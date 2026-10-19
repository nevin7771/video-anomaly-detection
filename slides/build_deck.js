// Builds the Second Review deck from the latest results.
//   cd slides && npm install pptxgenjs@3.12.0 && node build_deck.js
// Reads ../results/metrics/*.json and ../results/figures/dark/*.png
// (render dark figures with: VAD_THEME=dark python scripts/eda.py && VAD_THEME=dark python scripts/evaluate.py)
const fs = require("fs");
const path = require("path");
const pptxgen = require("pptxgenjs");

const ROOT = path.resolve(__dirname, "..");
const OUT = process.argv[2] || path.join(ROOT, "slides", "Video_Anomaly_Detection_Second_Review.pptx");
const FIG = (f) => path.join(ROOT, "results/figures/dark", f);
const summary = JSON.parse(fs.readFileSync(path.join(ROOT, "results/metrics/summary.json")));
const eda = JSON.parse(fs.readFileSync(path.join(ROOT, "results/metrics/eda_summary.json")));
const P = summary.primary;

// ---------- palette & type (matches the First Review deck) ----------
const BG = "0E1116", CARD = "151A21", CARD2 = "1A2029", LINE = "2A323F", LINE2 = "3A4250";
const INK = "EAEEF2", MUTED = "8A94A3", CYAN = "22D3EE", RED = "E66767", GREEN = "34D399", AMBER = "F5B44A";
const SANS = "Inter", MONO = "JetBrains Mono";
const W = 13.333, H = 7.5, MX = 0.55;
const SERIES = { pca: "C98500", "cae-nomotion": "199E70", cae: "3987E5", vit: "D95926" };
const LABEL = { pca: "PCA baseline", "cae-nomotion": "CAE (appearance only)", cae: "CAE + motion", vit: "Masked ViT-AE + motion" };

const pct = (x, d = 1) => (100 * x).toFixed(d);
const [PS, PP] = summary.primary_protocol || ["patch_max", "global"];
const tableRow = (v, score = PS, protocol = PP) =>
  summary.table.find((e) => e.variant === v && e.score === score && e.protocol === protocol);
const variants = ["pca", "cae-nomotion", "cae", "vit"].filter((v) => P[v]);
const best = variants.reduce((a, b) => (P[a].auc >= P[b].auc ? a : b));
const deep = ["cae", "vit"].filter((v) => P[v]);
const bestDeep = deep.reduce((a, b) => (tableRow(a).auc >= tableRow(b).auc ? a : b));

function pngSize(file) {
  const b = fs.readFileSync(file);
  return { w: b.readUInt32BE(16), h: b.readUInt32BE(20) };
}
// Fit an image inside a box, preserving aspect ratio, centred
function fitImage(slide, file, x, y, w, h, name) {
  const s = pngSize(file);
  const r = Math.min(w / s.w, h / s.h);
  const iw = s.w * r, ih = s.h * r;
  slide.addImage({ path: file, x: x + (w - iw) / 2, y: y + (h - ih) / 2, w: iw, h: ih, objectName: name });
}

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE";
pres.author = "Naveen (24EMDT7218)";
pres.title = "Video Anomaly Detection for Intelligent Surveillance — Second Review";
pres.theme = { headFontFace: SANS, bodyFontFace: SANS };

let TOTAL = 19;
let page = 0;

function txt(slide, text, o) {
  slide.addText(text, { fontFace: SANS, color: INK, margin: 0, isTextBox: true, valign: "top", ...o });
}
function mono(slide, text, o) {
  txt(slide, text, { fontFace: MONO, fontSize: 8, color: MUTED, charSpacing: 1.5, ...o });
}
function card(slide, x, y, w, h, o = {}) {
  slide.addShape(pres.shapes.RECTANGLE, { x, y, w, h, fill: { color: o.fill || CARD }, line: { color: o.line || LINE, width: 0.75 } });
}
function hline(slide, x, y, w, color = LINE) {
  slide.addShape(pres.shapes.LINE, { x, y, w, h: 0, line: { color, width: 0.75 } });
}

// Standard content slide chrome. rubric = [criterion, marks]
function content(section, kicker, titleRuns, subtitle, rubric) {
  page += 1;
  const s = pres.addSlide();
  s.background = { color: BG };
  mono(s, [
    { text: "VAD · SECOND REVIEW  /  ", options: { color: MUTED } },
    { text: section, options: { color: CYAN } },
  ], { x: MX, y: 0.32, w: 7, h: 0.25, fontSize: 7.5 });
  s.addShape(pres.shapes.OVAL, { x: W - MX - 2.25, y: 0.385, w: 0.07, h: 0.07, fill: { color: CYAN }, line: { color: CYAN } });
  mono(s, "REC · CAM-02 · 23.10.2026", { x: W - MX - 2.1, y: 0.32, w: 2.1, h: 0.25, fontSize: 7.5, align: "right" });
  mono(s, kicker, { x: MX, y: 0.82, w: 8, h: 0.25, color: CYAN });
  txt(s, titleRuns, { x: MX, y: 1.05, w: subtitle ? 7.6 : 12.2, h: 1.15, fontSize: 30, bold: true, valign: "top", paraSpaceAfter: 0, lineSpacingMultiple: 0.95 });
  if (subtitle) txt(s, subtitle, { x: 8.45, y: 1.2, w: W - MX - 8.45, h: 0.95, fontSize: 11.5, color: MUTED, align: "right", valign: "bottom" });
  hline(s, MX, 7.0, W - 2 * MX);
  if (rubric) {
    mono(s, [
      { text: "RUBRIC  ·  ", options: { color: MUTED } },
      { text: rubric[0].toUpperCase(), options: { color: CYAN } },
      { text: `  ·  ${rubric[1]} MARKS`, options: { color: MUTED } },
    ], { x: MX, y: 7.1, w: 8, h: 0.22, fontSize: 7 });
  }
  mono(s, `${String(page).padStart(2, "0")} / ${TOTAL}`, { x: W - MX - 1.5, y: 7.1, w: 1.5, h: 0.22, fontSize: 7, align: "right" });
  return s;
}
const T = (a, hl, b = "") => [{ text: a }, { text: hl, options: { color: CYAN } }, { text: b }];

// ======================= 01 COVER =======================
{
  page += 1;
  const s = pres.addSlide();
  s.background = { color: BG };
  mono(s, [
    { text: "MSC DATA SCIENCE  /  2026–27 BATCH  /  ", options: { color: CYAN } },
    { text: "SECOND REVIEW", options: { color: MUTED } },
  ], { x: MX, y: 0.5, w: 7, h: 0.25, fontSize: 7.5 });
  s.addShape(pres.shapes.OVAL, { x: W - MX - 2.25, y: 0.565, w: 0.07, h: 0.07, fill: { color: CYAN }, line: { color: CYAN } });
  mono(s, "REC · CAM-02 · 23.10.2026", { x: W - MX - 2.1, y: 0.5, w: 2.1, h: 0.25, fontSize: 7.5, align: "right" });
  mono(s, `01 / ${TOTAL} · COVER`, { x: MX, y: 1.55, w: 4, h: 0.25 });
  txt(s, [
    { text: "Video Anomaly", options: { breakLine: true } },
    { text: "Detection for", options: { breakLine: true } },
    { text: "Intelligent Surveillance.", options: { color: CYAN } },
  ], { x: MX, y: 1.8, w: 7.0, h: 2.45, fontSize: 40, bold: true, lineSpacingMultiple: 0.92 });
  txt(s, "From proposal to working pipeline: data acquired and verified, CAE and masked ViT-AE trained on normal-only footage, and first frame-level results on UCSD Ped2.",
    { x: MX, y: 4.4, w: 6.4, h: 0.9, fontSize: 13, color: MUTED });
  s.addShape(pres.shapes.RECTANGLE, { x: MX, y: 5.45, w: 1.65, h: 0.34, fill: { color: BG }, line: { color: CYAN, width: 0.75 } });
  mono(s, "SECOND REVIEW", { x: MX, y: 5.45, w: 1.65, h: 0.34, color: CYAN, align: "center", valign: "middle", fontSize: 8 });
  mono(s, "23 — 25 OCTOBER 2026", { x: MX + 1.8, y: 5.45, w: 3, h: 0.34, valign: "middle", fontSize: 8 });

  // Camera frame with real heat-map crop
  const fx = 7.65, fy = 1.75, fw = 5.13, fh = 3.6;
  s.addShape(pres.shapes.RECTANGLE, { x: fx, y: fy, w: fw, h: fh, fill: { color: "0B0E12" }, line: { color: CYAN, width: 1 } });
  fitImage(s, FIG("cover_frame.png"), fx + 0.12, fy + 0.35, fw - 0.24, fh - 0.7, "Cover heat-map frame");
  mono(s, "CAM-02 · UCSD PED2 · TEST004", { x: fx + 0.12, y: fy + 0.08, w: 3, h: 0.22, fontSize: 7, color: MUTED });
  mono(s, "LIVE", { x: fx + fw - 1.1, y: fy + 0.08, w: 1, h: 0.22, fontSize: 7, color: CYAN, align: "right" });
  mono(s, `${LABEL[bestDeep].toUpperCase()} · RECONSTRUCTION ERROR`, { x: fx + 0.12, y: fy + fh - 0.3, w: 4.2, h: 0.22, fontSize: 7, color: MUTED });
  mono(s, `AUC ${pct(tableRow(bestDeep).auc)}`, { x: fx + fw - 1.6, y: fy + fh - 0.3, w: 1.5, h: 0.22, fontSize: 7, color: CYAN, align: "right" });

  hline(s, MX, 6.2, W - 2 * MX);
  const cols = [["SUBMITTED BY", "Naveen", "Roll No. — 24EMDT7218"], ["SPECIALIZATION", "Data Science", "Computer Vision · Deep Learning"],
    ["GUIDE", "Praveen T", "Dept. of Computer Science"], ["REVIEW WINDOW", "23 — 25 Oct 2026", "Second Review · Panel Session"]];
  cols.forEach((c, i) => {
    const x = MX + i * 3.06;
    mono(s, c[0], { x, y: 6.35, w: 2.9, h: 0.2, fontSize: 6.5 });
    txt(s, c[1], { x, y: 6.58, w: 2.9, h: 0.3, fontSize: 12.5 });
    mono(s, c[2], { x, y: 6.9, w: 2.9, h: 0.2, fontSize: 6.5, charSpacing: 0.5 });
  });
  mono(s, `01 / ${TOTAL}`, { x: W - MX - 1.5, y: 7.15, w: 1.5, h: 0.2, fontSize: 7, align: "right" });
}

// ======================= 02 ROADMAP (rubric map) =======================
{
  const s = content("AGENDA", "02 / ROADMAP · MAPPED TO THE REVIEW RUBRIC", T("Six criteria, ", "one walk-through."),
    "Each section answers one rubric criterion; the footer of every slide names the criterion it evidences.", null);
  const rows = [
    ["01", "Literature Review & Theoretical Background", "Conceptual framework · 23 sources · evolution of methods · research gaps", "04 – 06", "8"],
    ["02", "Research Design / Project Plan", "Sampling plan · experimental design · tools · milestones", "07 – 08", "6"],
    ["03", "Data Collection Progress", "Acquisition · integrity verification · exploratory analysis", "09 – 10", "6"],
    ["04", "Analysis & Interpretation (Preliminary)", "Models built · frame-level AUC · ROC · localisation · ablations", "11 – 15", "8"],
    ["05", "Consistency with Objectives", "SMART objectives vs evidence · scope refinement · next steps", "16 – 17", "6"],
    ["06", "Presentation & Clarity", "Logical sequence · one message per slide · reproducible figures", "all", "6"],
  ];
  mono(s, "#", { x: MX, y: 2.45, w: 0.5, h: 0.2, fontSize: 7 });
  mono(s, "CRITERION", { x: 1.25, y: 2.45, w: 5, h: 0.2, fontSize: 7 });
  mono(s, "EVIDENCE", { x: 6.1, y: 2.45, w: 4.5, h: 0.2, fontSize: 7 });
  mono(s, "SLIDES", { x: 10.75, y: 2.45, w: 1, h: 0.2, fontSize: 7 });
  mono(s, "MARKS", { x: 11.75, y: 2.45, w: 1.03, h: 0.2, fontSize: 7, align: "right" });
  rows.forEach((r, i) => {
    const y = 2.72 + i * 0.68;
    hline(s, MX, y, W - 2 * MX);
    txt(s, r[0], { x: MX, y: y + 0.2, w: 0.5, h: 0.3, fontSize: 13, color: MUTED });
    txt(s, r[1], { x: 1.25, y: y + 0.19, w: 4.8, h: 0.3, fontSize: 14, bold: true });
    txt(s, r[2], { x: 6.1, y: y + 0.21, w: 4.5, h: 0.3, fontSize: 10.5, color: MUTED });
    txt(s, r[3], { x: 10.75, y: y + 0.2, w: 1, h: 0.3, fontSize: 12, fontFace: MONO, color: CYAN });
    txt(s, r[4], { x: 11.75, y: y + 0.17, w: 1.03, h: 0.35, fontSize: 16, bold: true, align: "right" });
  });
  hline(s, MX, 2.72 + 6 * 0.68, W - 2 * MX);
  mono(s, "TOTAL · 40 MARKS", { x: W - MX - 3, y: 2.72 + 6 * 0.68 + 0.08, w: 3, h: 0.2, align: "right", color: CYAN });
}

// ======================= 03 PROGRESS SINCE REVIEW 1 =======================
{
  const s = content("STATUS", "03 / SINCE THE FIRST REVIEW", T("From plan to a ", "working pipeline."),
    "Everything promised for the Review 2 gate is in place; Avenue and ComplexVAD scale-up move into the final month.",
    ["Consistency with Objectives", 6]);
  const done = [
    ["Literature review", "23 sources (Scopus/JCR venues: CVPR, ICCV, TPAMI, TIP, ACM CSUR)", "done"],
    ["Dataset acquired", "UCSD Ped2 downloaded, integrity-verified against official counts", "done"],
    ["Preprocessing pipeline", "Resize, 5-frame clips, motion channels, cached tensors", "done"],
    ["Baseline CAE", `Trained on normal clips · frame AUC ${pct(tableRow("cae").auc)} %`, "done"],
    ["Advanced ViT model", `Masked ViT-AE trained · frame AUC ${pct(tableRow("vit").auc)} %`, "prelim"],
    ["CUHK Avenue · ComplexVAD", "Same pipeline; Avenue next, ComplexVAD licence + subset", "next"],
  ];
  const stat = { done: ["DONE", GREEN], prelim: ["PRELIMINARY", AMBER], next: ["NEXT", MUTED] };
  done.forEach((d, i) => {
    const col = i % 2, row = Math.floor(i / 2);
    const x = MX + col * 6.2, y = 2.5 + row * 1.45, w = 6.0, h = 1.25;
    card(s, x, y, w, h);
    txt(s, d[0], { x: x + 0.3, y: y + 0.25, w: 3.8, h: 0.35, fontSize: 15, bold: true });
    txt(s, d[1], { x: x + 0.3, y: y + 0.68, w: 5.4, h: 0.4, fontSize: 11, color: MUTED });
    const [label, c] = stat[d[2]];
    s.addShape(pres.shapes.RECTANGLE, { x: x + w - 1.55, y: y + 0.27, w: 1.3, h: 0.28, fill: { color: BG }, line: { color: c, width: 0.75 } });
    mono(s, label, { x: x + w - 1.55, y: y + 0.27, w: 1.3, h: 0.28, color: c, align: "center", valign: "middle", fontSize: 7 });
  });
}

// ======================= 04 CONCEPTUAL FRAMEWORK =======================
{
  const s = content("01 · LITERATURE", "04 / THEORETICAL BACKGROUND", T("Anomaly = what a model of ", "normal", " cannot explain."),
    "One-class novelty detection (Chandola et al., 2009; Pang et al., 2021): learn p(x | normal) from normal footage only, score deviations.",
    ["Literature Review & Theoretical Background", 8]);
  const steps = [
    ["NORMAL-ONLY VIDEO", "Unlabelled footage of everyday scenes", "Training set"],
    ["REPRESENTATION LEARNER", "CAE / masked ViT-AE learns the normal manifold", "Hasan 2016 · He 2022"],
    ["RECONSTRUCTION ERROR", "‖x − x̂‖² per pixel → error map", "Liu 2018 · Gong 2019"],
    ["ANOMALY SCORE s(t)", "Patch-max, temporally smoothed", "Ramachandra 2022"],
    ["DECISION + LOCATION", "s(t) > τ → alert + heat map", "Sabokrou 2018"],
  ];
  const bw = 2.27, gap = 0.2, y = 2.55;
  steps.forEach((st, i) => {
    const x = MX + i * (bw + gap);
    card(s, x, y, bw, 1.75, { line: i === 1 ? CYAN : LINE });
    mono(s, `0${i + 1}`, { x: x + 0.2, y: y + 0.18, w: 0.5, h: 0.2, color: CYAN });
    txt(s, st[0], { x: x + 0.2, y: y + 0.45, w: bw - 0.4, h: 0.45, fontSize: 11.5, bold: true });
    txt(s, st[1], { x: x + 0.2, y: y + 0.92, w: bw - 0.4, h: 0.55, fontSize: 10, color: MUTED });
    mono(s, st[2], { x: x + 0.2, y: y + 1.45, w: bw - 0.3, h: 0.2, fontSize: 6.5, charSpacing: 0.5 });
    if (i < 4) txt(s, "›", { x: x + bw - 0.02, y: y + 0.6, w: 0.24, h: 0.4, fontSize: 18, color: CYAN, align: "center" });
  });
  mono(s, "THREE DESIGN QUESTIONS THAT ORGANISE THE LITERATURE — AND THIS PROJECT'S ANSWER", { x: MX, y: 4.6, w: 10, h: 0.2, fontSize: 7 });
  const qs = [
    ["What signal?", "Appearance · optical flow · objects · scene graphs", "Appearance + frame-difference motion"],
    ["Which learner?", "Sparse coding · CAE · ConvLSTM · GAN · memory-AE · transformers", "CAE vs masked ViT-AE, same pipeline"],
    ["How to decide?", "Global τ · per-video normalisation · patch / object level", "Patch-max score, τ from training scores"],
  ];
  qs.forEach((q, i) => {
    const y2 = 4.9 + i * 0.66;
    hline(s, MX, y2, W - 2 * MX);
    txt(s, q[0], { x: MX, y: y2 + 0.18, w: 2.2, h: 0.3, fontSize: 13, bold: true });
    txt(s, q[1], { x: 2.9, y: y2 + 0.2, w: 5.6, h: 0.3, fontSize: 10.5, color: MUTED });
    txt(s, q[2], { x: 8.6, y: y2 + 0.19, w: 4.2, h: 0.3, fontSize: 11.5, color: CYAN });
  });
}

// ======================= 05 EVOLUTION OF METHODS =======================
{
  const s = content("01 · LITERATURE", "05 / EVOLUTION OF METHODS · 2010 → 2025", T("Fifteen years, ", "six families", " of VAD."),
    "Each family fixed a weakness of the last. Reconstruction remains the dominant unsupervised paradigm; transformers and VLMs are the current frontier.",
    ["Literature Review & Theoretical Background", 8]);
  const fams = [
    ["2010–13", "Classical", "Dynamic textures, sparse coding; introduced UCSD Ped2 & Avenue", "Mahadevan · CVPR 2010\nLu · ICCV 2013"],
    ["2016–17", "Reconstruction", "Conv / ConvLSTM autoencoders learn temporal regularity", "Hasan · CVPR 2016\nLuo · ICME 2017"],
    ["2018", "Prediction & weak sup.", "Predict the future frame; MIL ranking on UCF-Crime", "Liu · CVPR 2018\nSultani · CVPR 2018"],
    ["2019–21", "Memory-augmented", "Memory of normal prototypes stops AEs reconstructing anomalies", "Gong · ICCV 2019\nPark · CVPR 2020\nLiu · ICCV 2021"],
    ["2021–24", "Self-sup. & masked", "Proxy tasks, masked auto-encoding, ViT tokens", "Georgescu · CVPR 2021\nHe · CVPR 2022\nRistea · CVPR 2024"],
    ["2024–25", "Vision-language", "CLIP-style semantics; complex interaction anomalies", "Wu · AAAI 2024\nMumcu · WACV-W 2025"],
  ];
  const y0 = 2.75, bw = 1.92, gap = 0.12;
  hline(s, MX, y0 + 0.42, W - 2 * MX, LINE2);
  fams.forEach((f, i) => {
    const x = MX + i * (bw + gap);
    s.addShape(pres.shapes.OVAL, { x: x, y: y0 + 0.36, w: 0.12, h: 0.12, fill: { color: i >= 4 ? CYAN : LINE2 }, line: { color: i >= 4 ? CYAN : LINE2 } });
    mono(s, f[0], { x, y: y0, w: bw, h: 0.22, color: i >= 4 ? CYAN : MUTED });
    card(s, x, y0 + 0.7, bw, 3.05, { fill: i === 4 ? CARD2 : CARD, line: i === 4 ? CYAN : LINE });
    txt(s, f[1], { x: x + 0.16, y: y0 + 0.88, w: bw - 0.3, h: 0.55, fontSize: 13, bold: true });
    txt(s, f[2], { x: x + 0.16, y: y0 + 1.5, w: bw - 0.3, h: 1.3, fontSize: 10, color: MUTED });
    mono(s, f[3], { x: x + 0.16, y: y0 + 2.6, w: bw - 0.25, h: 1.05, fontSize: 6.5, charSpacing: 0.3, color: INK, valign: "bottom" });
  });
  mono(s, "THIS PROJECT SITS IN THE HIGHLIGHTED FAMILY: CAE (RECONSTRUCTION) VS MASKED ViT-AE (SELF-SUPERVISED, MASKED)", { x: MX, y: 6.68, w: 12, h: 0.2, fontSize: 7, color: CYAN });
}

// ======================= 06 COMPARATIVE TABLE + GAPS =======================
{
  const s = content("01 · LITERATURE", "06 / COMPARATIVE REVIEW & RESEARCH GAPS", T("The field so far — ", "and its gaps."),
    "Published frame-level AUC (%), as reported by each paper. Ped2 is near-saturated; complex interaction anomalies are not.",
    ["Literature Review & Theoretical Background", 8]);
  const hdr = ["Study", "Venue", "Approach", "Ped2", "Avenue"];
  const rows = [
    ["Hasan et al. 2016", "CVPR", "Conv AE", "90.0", "70.2"],
    ["Luo et al. 2017", "ICME", "ConvLSTM-AE", "88.1", "77.0"],
    ["Liu et al. 2018", "CVPR", "Future-frame prediction", "95.4", "85.1"],
    ["Gong et al. 2019", "ICCV", "Memory-augmented AE", "94.1", "83.3"],
    ["Park et al. 2020", "CVPR", "Memory-guided (MNAD)", "97.0", "88.5"],
    ["Liu et al. 2021", "ICCV", "Hybrid flow (HF²-VAD)", "99.3", "91.1"],
  ];
  const opts = { fontFace: SANS, fontSize: 10.5, color: INK, valign: "middle", margin: [0, 0.08, 0, 0.08] };
  const data = [hdr.map((h) => ({ text: h.toUpperCase(), options: { ...opts, fontFace: MONO, fontSize: 7, color: MUTED, fill: { color: BG } } }))]
    .concat(rows.map((r) => r.map((c, j) => ({ text: c, options: { ...opts, fill: { color: BG }, bold: j >= 3, align: j >= 3 ? "right" : "left", color: j === 0 ? INK : (j >= 3 ? INK : MUTED) } }))));
  s.addTable(data, { x: MX, y: 2.45, w: 7.3, colW: [1.85, 0.9, 2.55, 1.0, 1.0], rowH: 0.42, border: { type: "solid", color: LINE, pt: 0.75 } });
  mono(s, "Values as reported by the original authors; protocols differ slightly between papers.", { x: MX, y: 5.55, w: 7.3, h: 0.3, fontSize: 6.5, charSpacing: 0.3 });

  const gaps = [
    ["Autoencoders generalise too well", "Plain AEs can rebuild anomalies (Gong 2019). Masking is a cheaper remedy than memory — tested here with a masked ViT-AE."],
    ["CNN vs transformer is rarely isolated", "Papers change inputs, losses and data at once. This project fixes the pipeline and swaps only the learner."],
    ["Small anomalies get averaged away", `Ped2 anomalies cover a median ${eda.anomaly_area_pct.median.toFixed(1)} % of the frame — frame-mean error dilutes them.`],
    ["Interaction anomalies are unsolved", "Best ComplexVAD frame AUC ≈ 0.60 (Mumcu 2025): a realistic target there is far below 0.90."],
  ];
  gaps.forEach((g, i) => {
    const y = 2.45 + i * 0.98;
    card(s, 8.15, y, W - MX - 8.15, 0.86);
    mono(s, `GAP 0${i + 1}`, { x: 8.35, y: y + 0.12, w: 1, h: 0.2, color: CYAN, fontSize: 7 });
    txt(s, g[0], { x: 9.25, y: y + 0.09, w: 3.4, h: 0.25, fontSize: 11.5, bold: true });
    txt(s, g[1], { x: 8.35, y: y + 0.38, w: W - MX - 8.55, h: 0.45, fontSize: 9, color: MUTED });
  });
}

// ======================= 07 RESEARCH DESIGN =======================
{
  const s = content("02 · RESEARCH DESIGN", "07 / SAMPLING PLAN & EXPERIMENTAL DESIGN", T("A controlled experiment: ", "change one thing at a time."),
    "Same data, same preprocessing, same scoring — only the learner (and, in the ablation, the motion input) changes.",
    ["Research Design / Project Plan", 6]);
  const cols = [
    ["SAMPLING PLAN", [
      ["Population", "Fixed-camera pedestrian-walkway CCTV (UCSD Ped2)"],
      ["Training sample", `${eda.train_videos} normal-only videos · ${eda.train_frames.toLocaleString()} frames`],
      ["Test sample", `${eda.test_videos} videos · ${eda.test_frames.toLocaleString()} frames · ${eda.test_anomalous_pct} % anomalous`],
      ["Unit of analysis", "5-frame clip centred on each frame"],
      ["Per epoch", "Random 50 % of clips (adjacent frames are near-duplicates)"],
    ]],
    ["EXPERIMENTAL DESIGN", [
      ["Independent var.", "Learner: PCA · CAE · masked ViT-AE"],
      ["Ablation", "Motion channels on / off (CAE)"],
      ["Controls", "128×192 input, clip length 5, σ = 3 smoothing"],
      ["Replication", "3 random seeds per deep model, mean ± sd"],
      ["Leakage guard", "No anomaly labels touch training or τ"],
    ]],
    ["EVALUATION PROTOCOL", [
      ["Primary metric", "Frame-level AUC-ROC (target > 0.90)"],
      ["Secondary", "EER · PR-AUC · precision / recall at τ"],
      ["Threshold τ", "99th percentile of training (normal) scores"],
      ["Uncertainty", "95 % bootstrap CI, resampling whole videos"],
      ["Significance", "Mann-Whitney U, Cohen's d"],
    ]],
  ];
  const cw = 3.95, gap = 0.18;
  cols.forEach((c, i) => {
    const x = MX + i * (cw + gap);
    card(s, x, 2.45, cw, 4.3, { line: i === 1 ? CYAN : LINE });
    mono(s, c[0], { x: x + 0.25, y: 2.65, w: cw - 0.5, h: 0.2, color: CYAN });
    c[1].forEach((r, j) => {
      const y = 3.0 + j * 0.73;
      hline(s, x + 0.25, y, cw - 0.5);
      txt(s, r[0], { x: x + 0.25, y: y + 0.1, w: cw - 0.5, h: 0.22, fontSize: 9, color: MUTED });
      txt(s, r[1], { x: x + 0.25, y: y + 0.32, w: cw - 0.5, h: 0.38, fontSize: 11 });
    });
  });
}

// ======================= 08 TOOLS + MILESTONES =======================
{
  const s = content("02 · PROJECT PLAN", "08 / TOOLS IDENTIFIED & MILESTONES", T("Four months, four gates — ", "on track."),
    "Updated Gantt: literature, data and baseline closed; the advanced model and second benchmark run into November.",
    ["Research Design / Project Plan", 6]);
  // Gantt: Aug 1 -> Nov 30
  const gx = MX + 2.6, gw = W - MX - gx, gy = 2.5;
  const day = (m, d) => ({ 8: 0, 9: 31, 10: 61, 11: 92 }[m] + d - 1);
  const X = (m, d) => gx + (day(m, d) / 122) * gw;
  ["AUG", "SEP", "OCT", "NOV"].forEach((mth, i) => {
    const x = X(8 + i, 1);
    mono(s, mth, { x: x + 0.05, y: gy, w: 1, h: 0.2, fontSize: 7 });
    s.addShape(pres.shapes.LINE, { x, y: gy + 0.25, w: 0, h: 3.05, line: { color: LINE, width: 0.5 } });
  });
  const ws = [
    ["Literature & scoping", [8, 1], [9, 20], "done"],
    ["Data & preprocessing", [9, 1], [10, 15], "done"],
    ["Baseline CAE", [9, 20], [10, 18], "done"],
    ["Masked ViT-AE", [10, 5], [11, 10], "active"],
    ["Avenue · ComplexVAD", [10, 25], [11, 15], "plan"],
    ["Experiments & thesis", [11, 1], [11, 26], "plan"],
  ];
  ws.forEach((w, i) => {
    const y = gy + 0.38 + i * 0.42;
    txt(s, w[0], { x: MX, y: y - 0.02, w: 2.5, h: 0.3, fontSize: 11 });
    const x1 = X(...w[1]), x2 = X(...w[2]);
    const fill = w[3] === "done" ? "3A4250" : w[3] === "active" ? CYAN : CARD2;
    s.addShape(pres.shapes.RECTANGLE, { x: x1, y, w: x2 - x1, h: 0.22, fill: { color: fill }, line: { color: w[3] === "plan" ? LINE2 : fill, width: 0.75 } });
  });
  const gates = [["R1", [8, 29]], ["R2 · NOW", [10, 24]], ["DRAFT", [11, 1]], ["VIVA", [11, 28]]];
  const yG = gy + 0.38 + 6 * 0.42;
  gates.forEach((g) => {
    const x = X(...g[1]);
    s.addShape(pres.shapes.DIAMOND, { x: x - 0.09, y: yG, w: 0.18, h: 0.18, fill: { color: g[0].includes("NOW") ? CYAN : BG }, line: { color: CYAN, width: 1 } });
    mono(s, g[0], { x: x - 0.6, y: yG + 0.22, w: 1.2, h: 0.2, fontSize: 6.5, align: "center", color: g[0].includes("NOW") ? CYAN : MUTED });
  });
  txt(s, "Review gates", { x: MX, y: yG - 0.04, w: 2.5, h: 0.3, fontSize: 11, color: MUTED });

  mono(s, "TOOLS IDENTIFIED — AND USED", { x: MX, y: 6.05, w: 4, h: 0.2, fontSize: 7 });
  const tools = ["Python 3.13", "PyTorch 2.x", "OpenCV", "NumPy · SciPy", "scikit-learn", "Matplotlib", "Git · GitHub", "Colab GPU (scale-up)"];
  let tx = MX;
  tools.forEach((t) => {
    const w = 0.22 + t.length * 0.085;
    s.addShape(pres.shapes.RECTANGLE, { x: tx, y: 6.33, w, h: 0.32, fill: { color: CARD }, line: { color: LINE2, width: 0.75 } });
    mono(s, t, { x: tx, y: 6.33, w, h: 0.32, align: "center", valign: "middle", color: INK, fontSize: 7.5, charSpacing: 0.5 });
    tx += w + 0.12;
  });
}

// ======================= 09 DATA ACQUISITION =======================
{
  const s = content("03 · DATA COLLECTION", "09 / ACQUISITION & VERIFICATION", T("Authentic, ", "verified", " benchmark data."),
    "Public, peer-reviewed benchmark with pixel-level ground truth; every count checked against the official release before any modelling.",
    ["Data Collection Progress", 6]);
  fitImage(s, FIG("eda_sample_frames.png"), MX, 2.4, W - 2 * MX, 1.75, "Sample frames");
  const checks = [
    ["Train videos", "16", "16"], ["Test videos", "12", "12"], ["Train frames", "2,550", "2,550"], ["Test frames", "2,010", "2,010"],
    ["Pixel masks", "2,010", "2,010"],
  ];
  mono(s, "INTEGRITY CHECK · scripts/prepare_data.py", { x: MX, y: 4.45, w: 6, h: 0.2, fontSize: 7 });
  mono(s, "OFFICIAL", { x: 3.3, y: 4.45, w: 1.1, h: 0.2, fontSize: 7, align: "right" });
  mono(s, "FOUND", { x: 4.5, y: 4.45, w: 1.0, h: 0.2, fontSize: 7, align: "right" });
  checks.forEach((c, i) => {
    const y = 4.7 + i * 0.42;
    hline(s, MX, y, 5.6);
    txt(s, c[0], { x: MX, y: y + 0.1, w: 2.6, h: 0.25, fontSize: 11 });
    txt(s, c[1], { x: 3.3, y: y + 0.1, w: 1.1, h: 0.25, fontSize: 11, color: MUTED, align: "right" });
    txt(s, c[2], { x: 4.5, y: y + 0.1, w: 1.0, h: 0.25, fontSize: 11, bold: true, align: "right" });
    txt(s, "✓", { x: 5.65, y: y + 0.08, w: 0.4, h: 0.25, fontSize: 12, color: GREEN });
  });
  const ds = [
    ["UCSD Ped2", "Mahadevan et al., CVPR 2010", "ACQUIRED · VERIFIED", GREEN, "460 MB · 28 videos · 240×360 · pixel masks"],
    ["CUHK Avenue", "Lu et al., ICCV 2013", "NEXT · SAME PIPELINE", AMBER, "37 videos · 640×360 · frame + pixel GT"],
    ["ComplexVAD", "Mumcu et al., WACV-W 2025 (MERL)", "LICENCE · SUBSET", MUTED, "39 GB · 34 h · 217 videos · 40 anomaly types"],
  ];
  ds.forEach((d, i) => {
    const y = 4.45 + i * 0.82;
    card(s, 6.75, y, W - MX - 6.75, 0.72);
    txt(s, d[0], { x: 6.95, y: y + 0.1, w: 2.4, h: 0.28, fontSize: 13, bold: true });
    mono(s, d[1], { x: 6.95, y: y + 0.42, w: 3.2, h: 0.2, fontSize: 6.5, charSpacing: 0.3 });
    mono(s, d[2], { x: 10.15, y: y + 0.12, w: 2.45, h: 0.2, fontSize: 7, color: d[3], align: "right" });
    txt(s, d[4], { x: 9.2, y: y + 0.4, w: 3.4, h: 0.25, fontSize: 8.5, color: MUTED, align: "right" });
  });
}

// ======================= 10 EDA =======================
{
  const me = eda.motion_energy;
  const s = content("03 · DATA COLLECTION", "10 / EXPLORATORY DATA ANALYSIS", T("What the data told us ", "before modelling."),
    "Two findings shaped the design: anomalies move differently, and they are small.",
    ["Data Collection Progress", 6]);
  fitImage(s, FIG("eda_motion_energy.png"), MX, 2.35, 6.0, 3.15, "Motion energy histogram");
  fitImage(s, FIG("eda_anomaly_area.png"), 6.8, 2.35, 6.0, 3.15, "Anomaly area histogram");
  const stats = [
    [`${me.cohens_d.toFixed(2)}`, "Cohen's d", `Motion energy, anomalous vs normal test frames · Mann-Whitney p < 10⁻${Math.floor(-Math.log10(me.p_value))}`, "→ add frame-difference motion channels"],
    [`${eda.anomaly_area_pct.median.toFixed(1)} %`, "median anomaly area", `90 % of anomalous regions cover < ${eda.anomaly_area_pct.p90.toFixed(1)} % of the frame`, "→ score with patch-max, not frame mean"],
    [`${eda.fully_anomalous_test_videos.length} of ${eda.test_videos}`, "test videos 100 % anomalous", `${eda.test_anomalous_pct} % of test frames are anomalous, which inflates PR-AUC and breaks per-video normalisation`, "→ AUC-ROC on global scores as primary"],
  ];
  stats.forEach((st, i) => {
    const x = MX + i * 4.13, y = 5.6;
    txt(s, st[0], { x, y, w: 1.65, h: 0.5, fontSize: 24, bold: true, color: CYAN });
    mono(s, st[1].toUpperCase(), { x: x + 1.7, y: y + 0.17, w: 2.2, h: 0.2, fontSize: 6.5 });
    txt(s, st[2], { x, y: y + 0.55, w: 3.95, h: 0.42, fontSize: 9, color: MUTED });
    txt(s, st[3], { x, y: y + 1.0, w: 3.95, h: 0.25, fontSize: 10, bold: true });
  });
}

// ======================= 11 MODELS AS BUILT =======================
{
  const s = content("04 · ANALYSIS", "11 / THE PIPELINE AS BUILT", T("Two learners, ", "one pipeline."),
    "Both models see the same 9-channel clip and are scored identically; only the representation learner differs.",
    ["Analysis & Interpretation (Preliminary)", 8]);
  // input strip
  card(s, MX, 2.45, 2.65, 3.9);
  mono(s, "INPUT CLIP", { x: MX + 0.2, y: 2.62, w: 2.3, h: 0.2, color: CYAN });
  [["5 frames", "grayscale · 128×192"], ["4 motion maps", "|frame t − frame t−1| × 4"], ["9 channels", "centred on scored frame"]].forEach((r, i) => {
    txt(s, r[0], { x: MX + 0.2, y: 3.05 + i * 0.95, w: 2.3, h: 0.35, fontSize: 15, bold: true });
    txt(s, r[1], { x: MX + 0.2, y: 3.42 + i * 0.95, w: 2.3, h: 0.3, fontSize: 9.5, color: MUTED });
  });
  const m = [
    ["cae", "Convolutional Autoencoder", "BASELINE · Hasan et al. 2016",
      ["3 conv stages, 24→48→96 channels, max-pool ×3", "64-channel 16×24 bottleneck, no skip connections", "Symmetric transposed-conv decoder, sigmoid output", "Loss: MSE on every pixel of normal clips"]],
    ["vit", "Masked ViT Autoencoder", "ADVANCED · He et al. 2022 · Ristea et al. 2024",
      ["16×16 patches → 96 tokens, dim 192, 4 layers, 4 heads", "Train: hide random 50 % of tokens, inpaint them", "Test: two checkerboard masks → every patch predicted while hidden", "Loss: MSE on hidden patches only"]],
  ];
  m.forEach((mm, i) => {
    const x = 3.4 + i * 4.75, w = 4.55;
    const p = P[mm[0]];
    card(s, x, 2.45, w, 3.9, { line: SERIES[mm[0]] });
    mono(s, mm[2], { x: x + 0.25, y: 2.62, w: w - 0.5, h: 0.2, color: CYAN, fontSize: 6.5, charSpacing: 0.5 });
    txt(s, mm[1], { x: x + 0.25, y: 2.92, w: w - 0.5, h: 0.4, fontSize: 16, bold: true });
    mm[3].forEach((b, j) => {
      txt(s, b, { x: x + 0.25, y: 3.45 + j * 0.42, w: w - 0.5, h: 0.4, fontSize: 10, color: INK, bullet: { indent: 12 } });
    });
    const st = [[`${(p.params / 1e6).toFixed(2)} M`, "PARAMETERS"], [`${p.fps_cpu.toFixed(0)}`, "FPS · 4-CORE CPU"], [`${p.epochs}`, "EPOCHS"]];
    st.forEach((q, k) => {
      txt(s, q[0], { x: x + 0.25 + k * 1.4, y: 5.3, w: 1.35, h: 0.4, fontSize: 17, bold: true });
      mono(s, q[1], { x: x + 0.25 + k * 1.4, y: 5.72, w: 1.4, h: 0.2, fontSize: 6 });
    });
  });
  mono(s, "SCORING (BOTH)  ·  squared-error map → max over 16×16 patches → Gaussian smoothing σ = 3 frames → τ = P99 of training scores",
    { x: MX, y: 6.55, w: 12.2, h: 0.2, fontSize: 7, color: INK });
}

// ======================= 12 HEADLINE RESULTS =======================
{
  const bR = tableRow(best);
  const s = content("04 · ANALYSIS", "12 / PRELIMINARY RESULTS · UCSD PED2", T("Best frame-level AUC: ", `${pct(bR.auc)} %.`),
    `${LABEL[best]}. Patch-max score, one global threshold across all test videos, mean of seeds; CI = 95 % bootstrap over videos.`,
    ["Analysis & Interpretation (Preliminary)", 8]);
  const labels = variants.map((v) => LABEL[v]);
  const vals = variants.map((v) => +(100 * tableRow(v).auc).toFixed(1));
  s.addChart(pres.charts.BAR, [{ name: "AUC-ROC (%)", labels, values: vals }], {
    x: MX, y: 2.4, w: 6.3, h: 4.35, barDir: "bar", chartColors: variants.map((v) => SERIES[v]),
    catAxisLabelColor: INK, catAxisLabelFontFace: "+mn-lt", catAxisLabelFontSize: 10, catAxisOrientation: "maxMin",
    valAxisLabelColor: MUTED, valAxisLabelFontFace: "+mn-lt", valAxisLabelFontSize: 8, valAxisMinVal: 50, valAxisMaxVal: 100, valAxisMajorUnit: 10,
    valGridLine: { color: LINE, size: 0.5 }, catGridLine: { style: "none" }, valAxisLineShow: false, catAxisLineShow: false,
    showValue: true, dataLabelPosition: "outEnd", dataLabelColor: INK, dataLabelFontSize: 11, dataLabelFontBold: true, dataLabelFontFace: "+mn-lt",
    dataLabelFormatCode: "0.0", showLegend: false, barGapWidthPct: 60,
    showTitle: true, title: "Frame-level AUC-ROC (%) · target 90", titleColor: MUTED, titleFontSize: 10, titleFontFace: "+mn-lt",
  });
  // table
  const hdr = ["MODEL", "AUC", "± SD", "EER", "PR-AUC", "95 % CI"].map((h) => ({ text: h, options: { fontFace: MONO, fontSize: 7, color: MUTED, fill: { color: BG }, align: "right" } }));
  hdr[0].options.align = "left";
  const rows = variants.map((v) => {
    const e = tableRow(v), ci = P[v].auc_ci95;
    const o = { fontFace: SANS, fontSize: 10.5, color: INK, fill: { color: v === best ? CARD2 : BG }, align: "right", valign: "middle" };
    return [
      { text: LABEL[v], options: { ...o, align: "left", bold: v === best } },
      { text: pct(e.auc), options: { ...o, bold: true, color: v === best ? CYAN : INK } },
      { text: e.n_seeds > 1 ? pct(e.auc_std) : "—", options: { ...o, color: MUTED } },
      { text: pct(e.eer), options: o },
      { text: pct(e.pr_auc), options: o },
      { text: `${pct(ci[0], 0)}–${pct(ci[1], 0)}`, options: { ...o, color: MUTED } },
    ];
  });
  s.addTable([hdr, ...rows], { x: 7.1, y: 2.5, w: W - MX - 7.1, colW: [2.15, 0.65, 0.6, 0.6, 0.75, 0.93], rowH: 0.42, border: { type: "solid", color: LINE, pt: 0.75 }, margin: [0, 0.06, 0, 0.06] });
  const thr = P[bestDeep].threshold_p99_train;
  card(s, 7.1, 4.85, W - MX - 7.1, 1.9);
  mono(s, `UNSUPERVISED OPERATING POINT · ${LABEL[bestDeep].toUpperCase()}`, { x: 7.3, y: 5.0, w: 5.3, h: 0.2, color: CYAN, fontSize: 7 });
  [[pct(thr.precision, 0) + " %", "PRECISION"], [pct(thr.recall, 0) + " %", "RECALL"], [pct(thr.f1, 0) + " %", "F1"], [pct(thr.fpr, 0) + " %", "FALSE-ALARM RATE"]].forEach((q, k) => {
    txt(s, q[0], { x: 7.3 + k * 1.33, y: 5.35, w: 1.3, h: 0.45, fontSize: 20, bold: true });
    mono(s, q[1], { x: 7.3 + k * 1.33, y: 5.85, w: 1.3, h: 0.2, fontSize: 6 });
  });
  txt(s, "τ = 99th percentile of scores on normal training clips — chosen without any anomaly label; global (un-normalised) scores.", { x: 7.3, y: 6.15, w: 5.3, h: 0.5, fontSize: 9, color: MUTED });
}

// ======================= 13 ROC + TIMELINES =======================
{
  const s = content("04 · ANALYSIS", "13 / HOW THE SCORES BEHAVE", T("Scores rise when ", "anomalies enter", " the scene."),
    `Left: ROC of every model. Right: ${LABEL[best]} score over time with ground-truth anomalous frames shaded.`,
    ["Analysis & Interpretation (Preliminary)", 8]);
  fitImage(s, FIG("roc_curves.png"), MX, 2.35, 4.6, 4.45, "ROC curves");
  fitImage(s, FIG("score_timelines.png"), 5.35, 2.35, W - MX - 5.35, 4.45, "Score timelines");
}

// ======================= 14 LOCALISATION =======================
{
  const s = content("04 · ANALYSIS", "14 / SPATIAL LOCALISATION", T("Not just ", "when", " — also where."),
    "Per-pixel reconstruction error doubles as an explanation: the operator sees which region triggered the alert.",
    ["Analysis & Interpretation (Preliminary)", 8]);
  fitImage(s, FIG("localisation_heatmaps.png"), MX, 2.3, 8.3, 4.55, "Localisation heat maps");
  const notes = [
    ["Vehicles and cyclists peak", "The strongest error sits on the cart and bicycles, inside the cyan ground-truth outline; the background stays dark."],
    ["Pedestrians are the noise floor", "Walkers still leave weaker error. In Test007 that floor causes a false peak before the anomaly starts (slide 13): a case for object-level scoring."],
    ["Explainability objective", "Every alert carries where (heat map), when (score curve) and why (reconstruction error)."],
  ];
  notes.forEach((n, i) => {
    const y = 2.4 + i * 1.5;
    card(s, 9.1, y, W - MX - 9.1, 1.35);
    txt(s, n[0], { x: 9.3, y: y + 0.15, w: W - MX - 9.5, h: 0.3, fontSize: 12.5, bold: true });
    txt(s, n[1], { x: 9.3, y: y + 0.52, w: W - MX - 9.5, h: 0.75, fontSize: 9.5, color: MUTED });
  });
}

// ======================= 15 INTERPRETATION / ABLATIONS =======================
{
  const s = content("04 · ANALYSIS", "15 / INTERPRETATION & ABLATIONS", T("What the ", "numbers", " mean."),
    "Ablations isolate the effect of each design decision; all values are frame-level AUC (%).",
    ["Analysis & Interpretation (Preliminary)", 8]);
  // ablation table: rows = variants, cols = mean/per-video, patch/per-video, mean/global, patch/global
  const cols = [["mean", "per-video"], ["patch_max", "per-video"], ["mean", "global"], ["patch_max", "global"]];
  const hdr = [{ text: "MODEL", options: { fontFace: MONO, fontSize: 7, color: MUTED, fill: { color: BG } } }]
    .concat(cols.map((c) => ({ text: `${c[0] === "mean" ? "FRAME MEAN" : "PATCH MAX"}\n${c[1].toUpperCase()}`, options: { fontFace: MONO, fontSize: 6.5, color: MUTED, fill: { color: BG }, align: "right" } })));
  const rows = variants.map((v) => [{ text: LABEL[v], options: { fontFace: SANS, fontSize: 10.5, color: INK, fill: { color: BG } } }]
    .concat(cols.map((c) => {
      const e = tableRow(v, c[0], c[1]);
      return { text: pct(e.auc), options: { fontFace: SANS, fontSize: 11, color: c[0] === PS && c[1] === PP ? CYAN : INK, bold: true, align: "right", fill: { color: BG } } };
    })));
  s.addTable([hdr, ...rows], { x: MX, y: 2.45, w: 6.4, colW: [2.4, 1.0, 1.0, 1.0, 1.0], rowH: [0.5, 0.45, 0.45, 0.45, 0.45], border: { type: "solid", color: LINE, pt: 0.75 }, margin: [0, 0.06, 0, 0.06], valign: "middle" });
  mono(s, "GLOBAL = one threshold for all videos (primary, highlighted) · PER-VIDEO = min-max within each video (literature protocol; distorted on Ped2 because 4 test videos are 100 % anomalous)", { x: MX, y: 4.95, w: 6.4, h: 0.4, fontSize: 6.5, charSpacing: 0.3 });
  card(s, MX, 5.5, 6.4, 1.25);
  mono(s, "STABILITY ACROSS 3 SEEDS · AUC STANDARD DEVIATION", { x: MX + 0.2, y: 5.62, w: 6, h: 0.2, color: CYAN, fontSize: 7 });
  [["cae", "CAE + motion"], ["vit", "Masked ViT-AE"]].forEach((q, k) => {
    txt(s, `± ${pct(tableRow(q[0]).auc_std)}`, { x: MX + 0.2 + k * 2.0, y: 5.9, w: 1.9, h: 0.4, fontSize: 18, bold: true });
    mono(s, q[1].toUpperCase(), { x: MX + 0.2 + k * 2.0, y: 6.35, w: 1.9, h: 0.2, fontSize: 6 });
  });
  txt(s, "Differences smaller than ~1 pt are within run-to-run noise.", { x: MX + 4.15, y: 5.92, w: 2.1, h: 0.75, fontSize: 9, color: MUTED });

  const f = (v, sc = PS, pr = PP) => (P[v] ? 100 * tableRow(v, sc, pr).auc : NaN);
  const dMotion = f("cae") - f("cae-nomotion");
  const dPatch = f(bestDeep) - f(bestDeep, "mean");
  const dArch = f("vit") - f("cae");
  const dNorm = f(bestDeep, PS, "per-video") - f(bestDeep);
  const sign = (x) => (x >= 0 ? "+" : "−") + Math.abs(x).toFixed(1);
  const finds = [
    [`${sign(dMotion)} pts`, "Motion channels (CAE)", dMotion >= 0 ? "Frame differences help: anomalies are partly defined by how things move, as the EDA predicted." : "Motion channels did not help the CAE yet; appearance dominates on Ped2. Revisit with optical flow."],
    [`${sign(dPatch)} pts`, "Patch-max vs frame mean", dPatch >= 0 ? "Small anomalies are no longer averaged away by thousands of normal pixels." : "Frame-mean is competitive here; patch-max remains needed for localisation."],
    [`${sign(dArch)} pts`, "Masked ViT-AE vs CAE", Math.abs(dArch) < 1.5 ? `A tie at this stage (within ~1 pt); the ViT gets there with ${(P.vit.fps_cpu / P.cae.fps_cpu).toFixed(0)}× faster CPU inference.` : dArch > 0 ? "Masking forces prediction from context, so unusual objects can't be copied through." : "The CAE's convolutional prior wins on this small dataset, consistent with ViTs needing more data."],
    [`${sign(dNorm)} pts`, "Per-video normalisation", dNorm < 0 ? "Hurts on Ped2: in the 4 fully anomalous videos it stretches anomalous frames down to 0. Global scores are the honest number." : "Inflates AUC by hiding scale differences between cameras; the global number is what a deployed system would see."],
  ];
  finds.forEach((q, i) => {
    const y = 2.45 + i * 1.1;
    card(s, 7.3, y, W - MX - 7.3, 0.98);
    txt(s, q[0], { x: 7.45, y: y + 0.12, w: 1.5, h: 0.45, fontSize: 19, bold: true, color: CYAN });
    txt(s, q[1], { x: 9.0, y: y + 0.1, w: W - MX - 9.15, h: 0.25, fontSize: 11.5, bold: true });
    txt(s, q[2], { x: 9.0, y: y + 0.38, w: W - MX - 9.15, h: 0.58, fontSize: 9, color: MUTED });
  });
}

// ======================= 16 OBJECTIVES VS EVIDENCE =======================
{
  const s = content("05 · CONSISTENCY", "16 / OBJECTIVES → EVIDENCE", T("Every SMART objective, ", "checked."),
    "The Review 1 commitments, the evidence produced so far, and their status at the Review 2 gate.",
    ["Consistency with Objectives", 6]);
  const bR = tableRow(bestDeep);
  const rows = [
    ["S", "Unsupervised frame-level detection with spatial localisation", "Normal-only training; frame scores + heat maps (slides 13–14)", "MET", GREEN],
    ["M", "Frame-level AUC-ROC > 0.90", `${pct(bR.auc)} % on Ped2 (${LABEL[bestDeep]}, ${bR.n_seeds} seeds)` + (bR.auc > 0.9 ? "" : ` · ${(90 - 100 * bR.auc).toFixed(1)} pts to go, plan on slide 17`), bR.auc > 0.9 ? "MET · PED2" : "PARTIAL", bR.auc > 0.9 ? GREEN : AMBER],
    ["A", "Public data, proven architectures, modest compute", "Runs end-to-end on a 4-core CPU in ~2 h; GPU optional", "MET", GREEN],
    ["R", "Real-world surveillance relevance", "Alerts carry location + score; τ set without labels", "MET", GREEN],
    ["T", "Delivered Aug – Nov 2026", "Review 2 gate met on schedule; Avenue / ComplexVAD in Nov", "ON TRACK", CYAN],
    ["+", "CAE vs ViT head-to-head comparison", "Same pipeline, same scoring, 3 seeds each (slides 12, 15)", "PRELIMINARY", AMBER],
  ];
  mono(s, "OBJECTIVE", { x: 1.25, y: 2.4, w: 4, h: 0.2, fontSize: 7 });
  mono(s, "EVIDENCE AT REVIEW 2", { x: 6.0, y: 2.4, w: 4, h: 0.2, fontSize: 7 });
  mono(s, "STATUS", { x: 11.0, y: 2.4, w: 1.78, h: 0.2, fontSize: 7, align: "right" });
  rows.forEach((r, i) => {
    const y = 2.65 + i * 0.68;
    hline(s, MX, y, W - 2 * MX);
    txt(s, r[0], { x: MX, y: y + 0.13, w: 0.6, h: 0.4, fontSize: 20, bold: true });
    txt(s, r[1], { x: 1.25, y: y + 0.19, w: 4.6, h: 0.35, fontSize: 11.5 });
    txt(s, r[2], { x: 6.0, y: y + 0.19, w: 4.9, h: 0.35, fontSize: 10, color: MUTED });
    mono(s, r[3], { x: 10.9, y: y + 0.24, w: 1.88, h: 0.2, color: r[4], align: "right", fontSize: 7.5 });
  });
  hline(s, MX, 2.65 + 6 * 0.68, W - 2 * MX);
}

// ======================= 17 SCOPE REFINEMENT + NEXT STEPS =======================
{
  const s = content("05 · CONSISTENCY", "17 / SCOPE REFINEMENT & NEXT STEPS", T("Same objectives, ", "sharper scope."),
    "One evidence-based adjustment: the > 0.90 AUC target applies to Ped2 / Avenue; ComplexVAD is reported against its published baseline.",
    ["Consistency with Objectives", 6]);
  card(s, MX, 2.45, 5.9, 4.3, { line: CYAN });
  mono(s, "SCOPE REFINEMENT", { x: MX + 0.25, y: 2.62, w: 5, h: 0.2, color: CYAN });
  const sc = [
    ["Primary benchmarks", "UCSD Ped2 (done) → CUHK Avenue (next): the datasets the >0.90 target was set against."],
    ["ComplexVAD", "Best published frame AUC is 0.60 (Mumcu 2025). Evaluate on a subset; success = beating that baseline, not 0.90."],
    ["Unchanged", "Unsupervised setting · CAE vs ViT comparison · localisation · Aug–Nov timeline."],
  ];
  sc.forEach((q, i) => {
    const y = 3.0 + i * 1.2;
    txt(s, q[0], { x: MX + 0.25, y, w: 5.4, h: 0.3, fontSize: 13, bold: true });
    txt(s, q[1], { x: MX + 0.25, y: y + 0.35, w: 5.4, h: 0.75, fontSize: 10, color: MUTED });
  });
  mono(s, "NEXT STEPS → FINAL REVIEW", { x: 6.85, y: 2.45, w: 5, h: 0.2, fontSize: 7 });
  const next = [
    ["01", "Close the Ped2 gap to 0.90: GPU training, longer schedules, optical-flow input", "Oct 25 – Nov 2"],
    ["02", "Run the identical pipeline on CUHK Avenue", "Oct 28 – Nov 6"],
    ["03", "ViT temporal head (SRU++ / attention over clips) + memory module", "Nov 1 – 12"],
    ["04", "ComplexVAD subset: frame + region-level AUC vs published baseline", "Nov 8 – 18"],
    ["05", "Failure-case taxonomy, runtime budget, thesis chapters 4–5", "Nov 12 – 26"],
  ];
  next.forEach((q, i) => {
    const y = 2.75 + i * 0.6;
    hline(s, 6.85, y, W - MX - 6.85);
    txt(s, q[0], { x: 6.85, y: y + 0.15, w: 0.45, h: 0.3, fontSize: 11, color: CYAN, fontFace: MONO });
    txt(s, q[1], { x: 7.35, y: y + 0.13, w: 3.9, h: 0.4, fontSize: 10.5 });
    mono(s, q[2], { x: 11.2, y: y + 0.18, w: 1.58, h: 0.2, align: "right", fontSize: 7 });
  });
  hline(s, 6.85, 2.75 + 5 * 0.6, W - MX - 6.85);
  mono(s, "RISKS & MITIGATION", { x: 6.85, y: 6.0, w: 5, h: 0.2, fontSize: 7, color: AMBER });
  txt(s, "ComplexVAD size (39 GB) → frame sub-sampling + Colab GPU.  ViT data-hunger → masking + pre-trained patch embeddings.",
    { x: 6.85, y: 6.25, w: W - MX - 6.85, h: 0.5, fontSize: 9.5, color: MUTED });
}

// ======================= 18 REFERENCES =======================
{
  const s = content("REFERENCES", "18 / SELECTED REFERENCES · FULL LIST (23) IN docs/literature_review.md", T("Grounded in ", "peer-reviewed", " work."), null, ["Literature Review & Theoretical Background", 8]);
  const refs = [
    "Chandola, Banerjee & Kumar (2009). Anomaly detection: A survey. ACM Computing Surveys, 41(3).",
    "Mahadevan et al. (2010). Anomaly detection in crowded scenes. CVPR.",
    "Lu, Shi & Jia (2013). Abnormal event detection at 150 FPS in MATLAB. ICCV.",
    "Hasan et al. (2016). Learning temporal regularity in video sequences. CVPR.",
    "Sabokrou et al. (2017). Deep-Cascade: cascading 3D deep neural networks… IEEE Trans. Image Processing, 26(4).",
    "Sabokrou et al. (2018). Deep-anomaly: fully convolutional neural network… Computer Vision and Image Understanding, 172.",
    "Liu, Luo, Lian & Gao (2018). Future frame prediction for anomaly detection – a new baseline. CVPR.",
    "Sultani, Chen & Shah (2018). Real-world anomaly detection in surveillance videos. CVPR.",
    "Kiran, Thomas & Parakkal (2018). Deep learning methods for unsupervised VAD: an overview. Journal of Imaging, 4(2).",
    "Gong et al. (2019). Memorizing normality to detect anomaly (MemAE). ICCV.",
    "Park, Noh & Ham (2020). Learning memory-guided normality for anomaly detection. CVPR.",
    "Pang, Shen, Cao & van den Hengel (2021). Deep learning for anomaly detection: a review. ACM Computing Surveys, 54(2).",
    "Liu et al. (2021). Hybrid VAD via memory-augmented flow reconstruction and flow-guided prediction. ICCV.",
    "Dosovitskiy et al. (2021). An image is worth 16×16 words: Transformers for image recognition. ICLR.",
    "He et al. (2022). Masked autoencoders are scalable vision learners. CVPR.",
    "Ramachandra, Jones & Vatsavai (2022). A survey of single-scene video anomaly detection. IEEE TPAMI, 44(5).",
    "Ristea et al. (2024). Self-distilled masked auto-encoders are efficient video anomaly detectors. CVPR.",
    "Mumcu, Jones, Cherian & Yilmaz (2025). ComplexVAD: Detecting interaction anomalies in video. WACV Workshops.",
  ];
  refs.forEach((r, i) => {
    const col = i < 9 ? 0 : 1, row = i % 9;
    const x = MX + col * 6.2, y = 2.35 + row * 0.5;
    txt(s, String(i + 1).padStart(2, "0"), { x, y, w: 0.4, h: 0.4, fontSize: 9, color: CYAN, fontFace: MONO });
    txt(s, r, { x: x + 0.45, y, w: 5.6, h: 0.45, fontSize: 9, color: INK });
  });
}

// ======================= 19 THANK YOU =======================
{
  page += 1;
  const s = pres.addSlide();
  s.background = { color: BG };
  mono(s, "VAD · SECOND REVIEW", { x: MX, y: 0.32, w: 4, h: 0.25, fontSize: 7.5 });
  txt(s, "“", { x: MX, y: 1.2, w: 1, h: 1, fontSize: 60, color: CYAN, bold: true });
  txt(s, "A model of normal is only as good as the anomalies it refuses to rebuild.", { x: MX, y: 2.1, w: 9.5, h: 1.3, fontSize: 28, bold: true });
  mono(s, "— WORKING PREMISE, REVISITED", { x: MX, y: 3.5, w: 5, h: 0.2 });
  txt(s, [{ text: "Thank you.", options: { breakLine: true } }, { text: "Questions & discussion welcome.", options: { color: MUTED, fontSize: 16, bold: false } }],
    { x: MX, y: 4.3, w: 8, h: 1.2, fontSize: 30, bold: true });
  hline(s, MX, 6.2, W - 2 * MX);
  [["CANDIDATE", "Naveen", "24EMDT7218"], ["GUIDE", "Praveen T", "Dept. of Computer Science"], ["CODE", "github.com/nevin7771", "video-anomaly-detection"], ["PROGRAMME", "MSc Data Science", "Second Review · 2026–27"]].forEach((c, i) => {
    const x = MX + i * 3.06;
    mono(s, c[0], { x, y: 6.35, w: 2.9, h: 0.2, fontSize: 6.5 });
    txt(s, c[1], { x, y: 6.58, w: 2.9, h: 0.3, fontSize: 12.5 });
    mono(s, c[2], { x, y: 6.9, w: 2.9, h: 0.2, fontSize: 6.5, charSpacing: 0.5 });
  });
  mono(s, `${TOTAL} / ${TOTAL}`, { x: W - MX - 1.5, y: 7.15, w: 1.5, h: 0.2, fontSize: 7, align: "right" });
}

if (page !== TOTAL) console.warn(`Slide count ${page} != TOTAL ${TOTAL}; update TOTAL.`);
pres.writeFile({ fileName: OUT }).then((f) => console.log("Wrote", f));
