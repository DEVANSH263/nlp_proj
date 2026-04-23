from flask import Blueprint, make_response, abort
from flask_login import login_required, current_user
from io import BytesIO
from datetime import datetime
import textwrap

from models import Prediction

report_bp = Blueprint('report', __name__, url_prefix='/report')

# ── Colour palette ────────────────────────────────────────────────────────────
C_DARK_BLUE   = (18,  40,  76)
C_MED_BLUE    = (37,  84, 155)
C_LIGHT_BLUE  = (214, 230, 255)
C_ACCENT_BLUE = (96, 145, 220)
C_RED         = (180,  25,  25)
C_RED_LIGHT   = (255, 220, 220)
C_RED_BG      = (254, 235, 235)
C_GREEN       = (16,  115,  54)
C_GREEN_LIGHT = (195, 245, 210)
C_GREEN_BG    = (225, 252, 235)
C_WHITE       = (255, 255, 255)
C_OFF_WHITE   = (250, 251, 253)
C_LIGHT_GREY  = (243, 244, 246)
C_BORDER_GREY = (210, 215, 225)
C_BLACK       = (22,   22,  22)
C_SUBTEXT     = (110, 115, 130)
C_LABEL       = (85,   90, 105)


# ── Chart helpers ─────────────────────────────────────────────────────────────

def _pie_chart(hof, not_):
    """HOF vs NOT pie chart → BytesIO PNG, or None."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(4.8, 3.6), facecolor='#f8f9fc')
        total = hof + not_
        if total == 0:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                    fontsize=12, color='#888')
            ax.axis('off')
        else:
            sizes  = [hof, not_]
            labels = [f'HOF  ({hof})', f'NOT  ({not_})']
            colors = ['#b41919', '#10733a']
            exp    = (0.06, 0.04)
            wedges, texts, autos = ax.pie(
                sizes, labels=labels, colors=colors,
                autopct='%1.1f%%', explode=exp,
                startangle=90, textprops={'fontsize': 9.5, 'color': '#1a1a1a'}
            )
            for at in autos:
                at.set_color('white')
                at.set_fontweight('bold')
                at.set_fontsize(9)
        ax.set_title('HOF vs NOT Distribution',
                     fontsize=11, fontweight='bold', pad=10, color='#12284c')
        fig.tight_layout(pad=1.2)
        buf = BytesIO()
        fig.savefig(buf, format='png', dpi=120,
                    bbox_inches='tight', facecolor='#f8f9fc')
        plt.close(fig)
        buf.seek(0)
        return buf
    except ImportError:
        return None


def _confidence_bar_chart(predictions):
    """Horizontal bar of avg confidence per label → BytesIO PNG, or None."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import numpy as np

        hof_vals = [p.confidence for p in predictions if p.prediction == 'HOF']
        not_vals = [p.confidence for p in predictions if p.prediction == 'NOT']
        labels   = ['HOF', 'NOT']
        means    = [
            float(np.mean(hof_vals)) * 100 if hof_vals else 0,
            float(np.mean(not_vals)) * 100 if not_vals else 0,
        ]
        colors = ['#b41919', '#10733a']

        fig, ax = plt.subplots(figsize=(4.8, 1.8), facecolor='#f8f9fc')
        bars = ax.barh(labels, means, color=colors, height=0.45,
                       edgecolor='white', linewidth=1.2)
        for bar, v in zip(bars, means):
            ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
                    f'{v:.1f}%', va='center', fontsize=9.5, color='#22222a')
        ax.set_xlim(0, 110)
        ax.set_xlabel('Avg Confidence (%)', fontsize=9, color='#555')
        ax.set_title('Avg Confidence by Label',
                     fontsize=10, fontweight='bold', color='#12284c')
        ax.tick_params(labelsize=9)
        ax.spines[['top', 'right']].set_visible(False)
        fig.tight_layout(pad=1.0)
        buf = BytesIO()
        fig.savefig(buf, format='png', dpi=120,
                    bbox_inches='tight', facecolor='#f8f9fc')
        plt.close(fig)
        buf.seek(0)
        return buf
    except ImportError:
        return None


# ── Drawing primitives ────────────────────────────────────────────────────────

def _card_open(pdf, height_hint, pad=4):
    """
    Draw a white card rectangle at current Y.
    Returns (x, y, w) of inner content area.
    height_hint: approximate card height; use 0 to draw just top border line.
    """
    M   = pdf.l_margin
    W   = pdf.w - M * 2
    y   = pdf.get_y()
    pdf.set_fill_color(*C_WHITE)
    pdf.set_draw_color(*C_BORDER_GREY)
    if height_hint:
        pdf.rect(M, y, W, height_hint, style='FD')
    # Left accent stripe
    pdf.set_fill_color(*C_MED_BLUE)
    if height_hint:
        pdf.rect(M, y, 3, height_hint, style='F')
    pdf.set_xy(M + 3 + pad, y + pad)
    return M + 3 + pad, y, W - 3 - pad * 2


def _section_title(pdf, title, icon=''):
    """Bold section heading with coloured left border."""
    M    = pdf.l_margin
    W    = pdf.w - M * 2
    y    = pdf.get_y()
    h    = 9
    # Background strip
    pdf.set_fill_color(*C_LIGHT_BLUE)
    pdf.set_draw_color(*C_ACCENT_BLUE)
    pdf.rect(M, y, W, h, style='FD')
    # Left thick accent
    pdf.set_fill_color(*C_MED_BLUE)
    pdf.rect(M, y, 4, h, style='F')
    # Text
    pdf.set_xy(M + 7, y + 1.5)
    pdf.set_font('Helvetica', 'B', 9.5)
    pdf.set_text_color(*C_DARK_BLUE)
    label = f'{icon}  {title}' if icon else title
    pdf.cell(W - 10, 6, label)
    pdf.set_text_color(*C_BLACK)
    pdf.set_xy(M, y + h + 2)


def _kv(pdf, label, value, fill=False, label_w=70):
    """Zebra key-value row."""
    M = pdf.l_margin
    W = pdf.w - M * 2
    pdf.set_fill_color(*(C_LIGHT_GREY if fill else C_WHITE))
    pdf.set_x(M)
    pdf.set_font('Helvetica', 'B', 8.5)
    pdf.set_text_color(*C_LABEL)
    pdf.cell(label_w, 6, label, fill=fill)
    pdf.set_font('Helvetica', '', 8.5)
    pdf.set_text_color(*C_BLACK)
    pdf.cell(W - label_w, 6, str(value), fill=fill, new_x='LMARGIN', new_y='NEXT')


def _stat_card(pdf, x, y, w, h, icon, number, sublabel, num_color):
    """Single stat card with icon label + big number + sublabel."""
    pad = 3
    # Card bg
    pdf.set_fill_color(*C_OFF_WHITE)
    pdf.set_draw_color(*C_BORDER_GREY)
    pdf.rect(x, y, w - 3, h, style='FD')
    # Top colour bar
    pdf.set_fill_color(*num_color)
    pdf.rect(x, y, w - 3, 3, style='F')
    # Icon text (top label)
    pdf.set_xy(x, y + 4)
    pdf.set_font('Helvetica', 'B', 7.5)
    pdf.set_text_color(*C_SUBTEXT)
    pdf.cell(w - 3, 5, icon, align='C')
    # Big number
    pdf.set_xy(x, y + 9)
    pdf.set_font('Helvetica', 'B', 22)
    pdf.set_text_color(*num_color)
    pdf.cell(w - 3, 12, number, align='C')
    # Sub-label
    pdf.set_xy(x, y + 20)
    pdf.set_font('Helvetica', '', 7)
    pdf.set_text_color(*C_SUBTEXT)
    pdf.cell(w - 3, 5, sublabel, align='C')
    pdf.set_text_color(*C_BLACK)


def _highlight_strip(pdf, hof_pct):
    """Thin dynamic strip: green = safe session, red = risky."""
    M = pdf.l_margin
    W = pdf.w - M * 2
    color = C_RED if hof_pct > 50 else C_GREEN
    label = 'HIGH RISK SESSION' if hof_pct > 50 else 'SAFE SESSION'
    pdf.set_fill_color(*color)
    y = pdf.get_y()
    pdf.rect(M, y, W, 5.5, style='F')
    pdf.set_xy(M, y + 0.5)
    pdf.set_font('Helvetica', 'B', 7)
    pdf.set_text_color(*C_WHITE)
    pdf.cell(W, 4.5, label, align='C')
    pdf.set_text_color(*C_BLACK)
    pdf.ln(8)


def _insight_box(pdf, text, color=C_LIGHT_BLUE, border_color=C_MED_BLUE):
    """Light coloured insight box with border."""
    M = pdf.l_margin
    W = pdf.w - M * 2
    y = pdf.get_y()
    wrap_width = max(40, int((W - 12) / 2.1))
    wrapped = textwrap.wrap(text, width=wrap_width) or ['']
    lines = max(2, len(wrapped))
    h = lines * 5 + 8
    pdf.set_fill_color(*color)
    pdf.set_draw_color(*border_color)
    pdf.rect(M, y, W, h, style='FD')
    # Left accent
    pdf.set_fill_color(*border_color)
    pdf.rect(M, y, 3.5, h, style='F')
    pdf.set_xy(M + 6, y + 3)
    pdf.set_font('Helvetica', '', 8.5)
    pdf.set_text_color(*C_BLACK)
    pdf.multi_cell(W - 8, 5, text)
    pdf.set_xy(M, y + h + 3)


def _ensure_space(pdf, required_height):
    """Start a new page if the upcoming block would overflow the current page."""
    if pdf.get_y() + required_height > pdf.page_break_trigger:
        pdf.add_page()


# ── Full History Report ───────────────────────────────────────────────────────

def _build_full_pdf(user, predictions):
    try:
        from fpdf import FPDF
    except ImportError:
        return b'fpdf2 not installed. Run: pip install fpdf2'

    hof_list  = [p for p in predictions if p.prediction == 'HOF']
    not_list  = [p for p in predictions if p.prediction == 'NOT']
    norm_list = [p for p in predictions if p.normalized_text]
    total     = len(predictions)
    hof_pct   = len(hof_list) / total * 100 if total else 0
    avg_conf  = sum(p.confidence for p in predictions) / total * 100 if total else 0

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=18)
    pdf.add_page()
    pdf.set_margins(14, 14, 14)
    M     = 14
    PW    = pdf.w - M * 2

    # ── HEADER ───────────────────────────────────────────────────────────────
    # Dark band
    pdf.set_fill_color(*C_DARK_BLUE)
    pdf.set_text_color(*C_WHITE)
    pdf.set_font('Helvetica', 'B', 24)
    pdf.cell(0, 18, 'HateShield AI',
             new_x='LMARGIN', new_y='NEXT', fill=True, align='C')

    # Subtitle band
    pdf.set_fill_color(*C_MED_BLUE)
    pdf.set_font('Helvetica', '', 9.5)
    pdf.cell(0, 7, 'Multilingual Hate Speech Detection System  |  Prediction Report',
             new_x='LMARGIN', new_y='NEXT', fill=True, align='C')

    # Date band
    pdf.set_fill_color(*C_LIGHT_BLUE)
    pdf.set_text_color(*C_MED_BLUE)
    pdf.set_font('Helvetica', 'I', 8.5)
    pdf.cell(0, 6,
             f'Generated: {datetime.utcnow().strftime("%d %B %Y  %H:%M UTC")}',
             new_x='LMARGIN', new_y='NEXT', fill=True, align='C')

    pdf.set_text_color(*C_BLACK)
    pdf.ln(5)

    # Dynamic highlight strip
    _highlight_strip(pdf, hof_pct)

    # ── USER INFO ─────────────────────────────────────────────────────────────
    _section_title(pdf, 'USER INFORMATION', '[>]')
    _kv(pdf, 'Username',    user.username,                          fill=False)
    _kv(pdf, 'Email',       user.email,                             fill=True)
    _kv(pdf, 'Report Date', datetime.utcnow().strftime('%d %B %Y'), fill=False)
    pdf.ln(6)

    # ── STAT CARDS ────────────────────────────────────────────────────────────
    _section_title(pdf, 'SUMMARY STATISTICS', '[#]')
    box_w = PW / 4
    box_h = 28
    y0    = pdf.get_y()

    avg_conf_str = f'{avg_conf:.1f}%'
    _stat_card(pdf, M,             y0, box_w, box_h,
               'TOTAL', str(total), 'Analyses', C_MED_BLUE)
    _stat_card(pdf, M + box_w,     y0, box_w, box_h,
               'HOF', str(len(hof_list)), 'Hate/Offensive', C_RED)
    _stat_card(pdf, M + box_w*2,   y0, box_w, box_h,
               'NOT', str(len(not_list)), 'Safe Content', C_GREEN)
    _stat_card(pdf, M + box_w*3,   y0, box_w, box_h,
               'AVG CONF', avg_conf_str, 'Confidence', C_ACCENT_BLUE)

    pdf.set_xy(M, y0 + box_h + 4)
    _kv(pdf, 'HOF Percentage',           f'{hof_pct:.1f}%',                 fill=False)
    _kv(pdf, 'NOT Percentage',           f'{100-hof_pct:.1f}%',             fill=True)
    _kv(pdf, 'With Normalization',       str(len(norm_list)),               fill=False)
    _kv(pdf, 'Without Normalization',    str(total - len(norm_list)),        fill=True)
    confidence_label = (
        'High' if avg_conf >= 80 else 'Moderate' if avg_conf >= 60 else 'Low'
    )
    _kv(pdf, 'Confidence Level',
        f'{avg_conf:.1f}%  ({confidence_label})', fill=False)
    pdf.ln(6)

    # ── DISTRIBUTION CHARTS ───────────────────────────────────────────────────
    _ensure_space(pdf, 90)
    _section_title(pdf, 'DISTRIBUTION CHARTS', '[~]')

    chart_y = pdf.get_y()
    # Chart card background
    pdf.set_fill_color(*C_OFF_WHITE)
    pdf.set_draw_color(*C_BORDER_GREY)
    pdf.rect(M, chart_y, PW, 68, style='FD')

    pie_buf  = _pie_chart(len(hof_list), len(not_list))
    bar_buf  = _confidence_bar_chart(predictions)

    half_w = PW / 2 - 4
    if pie_buf:
        pdf.image(pie_buf,  x=M + 2,          y=chart_y + 2, w=half_w)
    if bar_buf:
        pdf.image(bar_buf,  x=M + half_w + 6, y=chart_y + 10, w=half_w - 2)

    if not pie_buf:
        pdf.set_xy(M + 4, chart_y + 30)
        pdf.set_font('Helvetica', 'I', 8)
        pdf.set_text_color(*C_SUBTEXT)
        pdf.cell(PW - 8, 6, 'Run: pip install matplotlib  to enable charts')
        pdf.set_text_color(*C_BLACK)

    pdf.set_xy(M, chart_y + 70)

    # Caption
    pdf.set_font('Helvetica', 'I', 7.5)
    pdf.set_text_color(*C_SUBTEXT)
    pdf.cell(0, 5,
             'Left: Label distribution   |   Right: Average confidence per label',
             new_x='LMARGIN', new_y='NEXT', align='C')
    pdf.set_text_color(*C_BLACK)
    pdf.ln(5)

    # ── NORMALIZATION INSIGHT ─────────────────────────────────────────────────
    _ensure_space(pdf, 40)
    _section_title(pdf, 'NORMALIZATION INSIGHT', '[i]')

    if norm_list:
        pct = len(norm_list) / total * 100
        text = (
            f'INSIGHT: Transliteration normalization was applied to {len(norm_list)} of '
            f'{total} predictions ({pct:.1f}%). The system mapped Romanized Hindi '
            '(Hinglish) tokens to English equivalents prior to classification. '
            'This technique reduces vocabulary mismatch caused by phonetic spelling '
            'variation, thereby strengthening hate-speech detection for code-mixed input.'
        )
    else:
        text = (
            'INSIGHT: No predictions utilized normalization in this session. '
            'This indicates that input text either did not contain significant '
            'transliteration variation, or normalization was not enabled by the user. '
            'Enabling normalization for Hinglish text can improve classification '
            'accuracy by resolving phonetic spelling inconsistencies.'
        )
    _insight_box(pdf, text, C_LIGHT_BLUE, C_MED_BLUE)

    # ── PREDICTIONS TABLE ─────────────────────────────────────────────────────
    _ensure_space(pdf, 28)
    _section_title(pdf, 'RECENT PREDICTIONS  (latest 20)', '[=]')

    col_w = [8, 70, 16, 20, 28]
    hdrs  = ['#', 'Input Text', 'Result', 'Conf.', 'Timestamp']

    # Header row
    pdf.set_fill_color(*C_DARK_BLUE)
    pdf.set_text_color(*C_WHITE)
    pdf.set_font('Helvetica', 'B', 8)
    for w, h in zip(col_w, hdrs):
        pdf.cell(w, 7, h, border=1, fill=True, align='C')
    pdf.ln()

    for idx, p in enumerate(predictions[:20], start=1):
        _ensure_space(pdf, 8)
        is_hof = p.prediction == 'HOF'
        row_bg = C_LIGHT_GREY if idx % 2 == 0 else C_WHITE

        raw   = p.input_text.encode('latin-1', errors='replace').decode('latin-1')
        short = (raw[:46] + '...') if len(raw) > 46 else raw
        ts    = p.timestamp.strftime('%d/%m/%y %H:%M')

        # Row number
        pdf.set_fill_color(*row_bg)
        pdf.set_text_color(*C_SUBTEXT)
        pdf.set_font('Helvetica', '', 7.5)
        pdf.cell(col_w[0], 6, str(idx), border=1, fill=True, align='C')

        # Text
        pdf.set_text_color(*C_BLACK)
        pdf.cell(col_w[1], 6, short, border=1, fill=True)

        # Coloured result badge
        pdf.set_fill_color(*(C_RED_BG   if is_hof else C_GREEN_BG))
        pdf.set_text_color(*(C_RED      if is_hof else C_GREEN))
        pdf.set_font('Helvetica', 'B', 8)
        pdf.cell(col_w[2], 6, p.prediction, border=1, fill=True, align='C')

        # Confidence + timestamp
        pdf.set_fill_color(*row_bg)
        pdf.set_text_color(*C_BLACK)
        pdf.set_font('Helvetica', '', 7.5)
        pdf.cell(col_w[3], 6, f'{p.confidence*100:.1f}%', border=1, fill=True, align='C')
        pdf.cell(col_w[4], 6, ts,                         border=1, fill=True, align='C')
        pdf.ln()

    pdf.ln(6)

    # ── CONCLUSION ────────────────────────────────────────────────────────────
    _section_title(pdf, 'FINAL OBSERVATION', '[*]')

    if total == 0:
        conclusion = ('No predictions have been recorded yet. '
                      'Submit text through the dashboard to generate analysis data.')
    elif hof_pct == 0:
        conclusion = (
            f'All {total} analyzed inputs were classified as non-offensive (NOT). '
            'No hate speech patterns were detected in this session. '
            'The system maintained stable performance with an average confidence of '
            f'{avg_conf:.1f}%. Session risk level: LOW.'
        )
    elif hof_pct == 100:
        conclusion = (
            f'All {total} analyzed inputs were classified as hate/offensive (HOF). '
            f'Average confidence: {avg_conf:.1f}%. Session risk level: HIGH. '
            'Consider reviewing moderation policies for the monitored content.'
        )
    else:
        dominant = 'NOT' if len(not_list) >= len(hof_list) else 'HOF'
        conclusion = (
            f'Out of {total} predictions, {len(hof_list)} were classified as HOF '
            f'({hof_pct:.1f}%) and {len(not_list)} as NOT '
            f'({100-hof_pct:.1f}%). '
            f'The dominant classification is {dominant}. '
            f'Average model confidence: {avg_conf:.1f}%. '
            f'Session risk level: {"MODERATE" if hof_pct < 50 else "HIGH"}.'
        )
    _insight_box(pdf, conclusion, C_GREEN_BG, C_GREEN)

    # ── FOOTER ────────────────────────────────────────────────────────────────
    pdf.ln(2)
    pdf.set_fill_color(*C_DARK_BLUE)
    pdf.set_text_color(*C_WHITE)
    pdf.set_font('Helvetica', 'I', 7.5)
    pdf.cell(0, 8,
             'HateShield AI  |  Multilingual Hate Speech Detection  |  '
             'Confidential Research Report',
             new_x='LMARGIN', new_y='NEXT', fill=True, align='C')

    return bytes(pdf.output())


# ── Single Prediction Report ──────────────────────────────────────────────────

def _build_single_pdf(user, pred):
    """One-page premium report for a single prediction."""
    try:
        from fpdf import FPDF
    except ImportError:
        return b'fpdf2 not installed.'

    is_hof   = pred.prediction == 'HOF'
    conf_pct = pred.confidence * 100
    pal_main = C_RED if is_hof else C_GREEN
    pal_bg   = C_RED_BG if is_hof else C_GREEN_BG

    pdf = FPDF()
    pdf.add_page()
    pdf.set_margins(14, 14, 14)
    M  = 14
    PW = pdf.w - M * 2

    # Header
    pdf.set_fill_color(*C_DARK_BLUE)
    pdf.set_text_color(*C_WHITE)
    pdf.set_font('Helvetica', 'B', 20)
    pdf.cell(0, 15, 'HateShield AI',
             new_x='LMARGIN', new_y='NEXT', fill=True, align='C')
    pdf.set_fill_color(*C_MED_BLUE)
    pdf.set_font('Helvetica', '', 9)
    pdf.cell(0, 6, 'Single Prediction Analysis Report',
             new_x='LMARGIN', new_y='NEXT', fill=True, align='C')
    pdf.set_fill_color(*C_LIGHT_BLUE)
    pdf.set_text_color(*C_MED_BLUE)
    pdf.set_font('Helvetica', 'I', 8)
    pdf.cell(0, 5.5,
             f'Generated: {datetime.utcnow().strftime("%d %B %Y  %H:%M UTC")}',
             new_x='LMARGIN', new_y='NEXT', fill=True, align='C')
    pdf.set_text_color(*C_BLACK)
    pdf.ln(6)

    # Verdict banner
    verdict = 'HATE / OFFENSIVE CONTENT DETECTED' if is_hof else 'CONTENT IS SAFE (NOT OFFENSIVE)'
    pdf.set_fill_color(*pal_main)
    pdf.set_text_color(*C_WHITE)
    pdf.set_font('Helvetica', 'B', 13)
    pdf.cell(0, 12, verdict, new_x='LMARGIN', new_y='NEXT', fill=True, align='C')
    pdf.set_text_color(*C_BLACK)
    pdf.ln(5)

    # Confidence gauge (visual bar)
    _section_title(pdf, 'CONFIDENCE SCORE', '[%]')
    y_bar = pdf.get_y()
    bar_h = 10
    # Background
    pdf.set_fill_color(*C_LIGHT_GREY)
    pdf.set_draw_color(*C_BORDER_GREY)
    pdf.rect(M, y_bar, PW, bar_h, style='FD')
    # Fill
    pdf.set_fill_color(*pal_main)
    fill_w = PW * pred.confidence
    pdf.rect(M, y_bar, fill_w, bar_h, style='F')
    # Label
    pdf.set_xy(M, y_bar + 1.5)
    pdf.set_font('Helvetica', 'B', 9)
    pdf.set_text_color(*C_WHITE)
    pdf.cell(fill_w, 7, f'  {conf_pct:.1f}%  Confidence', align='L')
    pdf.set_text_color(*C_BLACK)
    pdf.set_xy(M, y_bar + bar_h + 3)

    conf_label = 'High confidence' if conf_pct >= 80 else 'Moderate confidence' if conf_pct >= 60 else 'Low confidence'
    pdf.set_font('Helvetica', 'I', 8)
    pdf.set_text_color(*C_SUBTEXT)
    pdf.cell(0, 5, f'{conf_label} -- model is {"strongly" if conf_pct>=80 else "moderately"} certain of this classification.',
             new_x='LMARGIN', new_y='NEXT')
    pdf.set_text_color(*C_BLACK)
    pdf.ln(5)

    # Input + Normalized text
    _section_title(pdf, 'INPUT TEXT', '[T]')
    raw_text = pred.input_text.encode('latin-1', errors='replace').decode('latin-1')
    _insight_box(pdf, raw_text, C_OFF_WHITE, C_BORDER_GREY)

    if pred.normalized_text:
        _section_title(pdf, 'NORMALIZED TEXT', '[N]')
        norm_raw = pred.normalized_text.encode('latin-1', errors='replace').decode('latin-1')
        _insight_box(pdf, norm_raw, C_LIGHT_BLUE, C_MED_BLUE)

    # Details
    _section_title(pdf, 'ANALYSIS DETAILS', '[i]')
    _kv(pdf, 'User',              user.username)
    _kv(pdf, 'Prediction',        pred.prediction,                          fill=True)
    _kv(pdf, 'Confidence',        f'{conf_pct:.2f}%')
    _kv(pdf, 'Normalization Used', 'Yes' if pred.normalized_text else 'No', fill=True)
    _kv(pdf, 'Timestamp',         pred.timestamp.strftime('%d %B %Y  %H:%M UTC'))
    pdf.ln(5)

    # Interpretation
    _section_title(pdf, 'INTERPRETATION', '[*]')
    if is_hof:
        interp = (
            f'The input text was classified as Hate/Offensive (HOF) with '
            f'{conf_pct:.1f}% confidence. The system identified patterns '
            'consistent with hate speech or offensive language. '
            'Review and moderate this content as appropriate.'
        )
    else:
        interp = (
            f'The input text was classified as Non-Offensive (NOT) with '
            f'{conf_pct:.1f}% confidence. No hate speech patterns were '
            'detected. The content appears safe for the monitored context.'
        )
    _insight_box(pdf, interp, pal_bg, pal_main)

    # Footer
    pdf.set_fill_color(*C_DARK_BLUE)
    pdf.set_text_color(*C_WHITE)
    pdf.set_font('Helvetica', 'I', 7.5)
    pdf.cell(0, 8,
             'HateShield AI  |  Single Prediction Report  |  Confidential',
             new_x='LMARGIN', new_y='NEXT', fill=True, align='C')

    return bytes(pdf.output())


# ── Routes ────────────────────────────────────────────────────────────────────

@report_bp.route('/download')
@login_required
def download_report():
    """Full history PDF report for current user."""
    predictions = (
        Prediction.query
        .filter_by(user_id=current_user.id)
        .order_by(Prediction.timestamp.desc())
        .all()
    )
    pdf_bytes = _build_full_pdf(current_user, predictions)
    filename  = (f'hateshield_{current_user.username}_'
                 f'{datetime.utcnow().strftime("%Y%m%d")}.pdf')
    resp = make_response(pdf_bytes)
    resp.headers['Content-Type']        = 'application/pdf'
    resp.headers['Content-Disposition'] = f'attachment; filename="{filename}"'
    return resp


@report_bp.route('/single/<int:pred_id>')
@login_required
def single_report(pred_id):
    """One-page PDF for a single prediction (must belong to current user)."""
    pred = Prediction.query.filter_by(
        id=pred_id, user_id=current_user.id
    ).first_or_404()

    pdf_bytes = _build_single_pdf(current_user, pred)
    filename  = f'hateshield_pred_{pred_id}.pdf'
    resp = make_response(pdf_bytes)
    resp.headers['Content-Type']        = 'application/pdf'
    resp.headers['Content-Disposition'] = f'attachment; filename="{filename}"'
    return resp
