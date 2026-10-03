const MODEL_URL = "https://github.com/ozzzzj/Decoder-DeepONet/releases/download/DDON-WEB/DDON.onnx";
let rows = [], prediction = null, session = null;
const statusEl = document.getElementById("status");

async function getSession() {
  if (session) return session;
  statusEl.textContent = "Loading DDON model (first use may take a moment)...";
  ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.23.2/dist/";
  session = await ort.InferenceSession.create(MODEL_URL, { executionProviders: ["wasm"] });
  return session;
}

document.getElementById("file").addEventListener("change", async e => {
  const text = await e.target.files[0].text();
  rows = text.split(/\r?\n/).filter(Boolean).map(line =>
    line.trim().split(/[\s,;]+/).map(Number)
  );
  statusEl.textContent = rows.length + " rows loaded.";
});

document.getElementById("run").addEventListener("click", async () => {
  if (rows.length !== 109 || rows.some(r => r.length < 2 || !Number.isFinite(r[0]) || !Number.isFinite(r[1]))) {
    statusEl.textContent = "Input must contain exactly 109 valid rows with two columns.";
    return;
  }
  try {
    const s = await getSession();
    statusEl.textContent = "Running DDON in your browser...";

    const flat = new Float32Array(109 * 2);
    rows.forEach((r, i) => { flat[i * 2] = r[0]; flat[i * 2 + 1] = r[1]; });

    const rawU = Number(document.getElementById("u").value);
    if (!Number.isFinite(rawU)) throw new Error("u must be a valid number.");
    const uNorm = new Float32Array([rawU / -1.0]);

    const valuesTensor = new ort.Tensor("float32", flat, [1, 109, 2]);
    const uTensor = new ort.Tensor("float32", uNorm, [1, 1]);

    const names = s.inputNames;
    if (names.length !== 2) throw new Error("Unexpected DDON model input count.");

    const feeds = {};
    feeds[names[0]] = valuesTensor;
    feeds[names[1]] = uTensor;

    const results = await s.run(feeds);
    const output = results[s.outputNames[0]];
    const efield = Array.from(output.data);

    prediction = { model: "ddon", z: rows.map(r => r[0]), efield };
    draw(prediction.z, prediction.efield);
    document.getElementById("result").hidden = false;
    statusEl.textContent = "Prediction complete (computed locally in your browser).";
  } catch (err) {
    console.error(err);
    statusEl.textContent = "Error: " + err.message;
  }
});

function draw(x, y) {
  const c = document.getElementById("plot"), ctx = c.getContext("2d");
  const p = 45, w = c.width - 2*p, h = c.height - 2*p;
  ctx.clearRect(0, 0, c.width, c.height);
  ctx.strokeStyle = "#222"; ctx.lineWidth = 1;
  ctx.beginPath(); ctx.moveTo(p,p); ctx.lineTo(p,p+h); ctx.lineTo(p+w,p+h); ctx.stroke();
  const xmin=Math.min(...x), xmax=Math.max(...x), ymin=Math.min(...y), ymax=Math.max(...y), dy=(ymax-ymin)||1;
  ctx.strokeStyle="#1769aa"; ctx.lineWidth=2; ctx.beginPath();
  y.forEach((v,i)=>{
    const px=p+(x[i]-xmin)/(xmax-xmin||1)*w, py=p+h-(v-ymin)/dy*h;
    i ? ctx.lineTo(px,py) : ctx.moveTo(px,py);
  });
  ctx.stroke();
  ctx.fillStyle="#222"; ctx.font="14px sans-serif";
  ctx.fillText("Normalized z", c.width/2-35, c.height-8);
  ctx.save(); ctx.translate(15,c.height/2+40); ctx.rotate(-Math.PI/2);
  ctx.fillText("Predicted E-field",0,0); ctx.restore();
}

document.getElementById("download").addEventListener("click", () => {
  if (!prediction) return;
  const lines = ["z,efield", ...prediction.z.map((z,i)=>z+","+prediction.efield[i])];
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([lines.join("\n")], {type:"text/csv"}));
  a.download = "ddon_prediction.csv"; a.click(); URL.revokeObjectURL(a.href);
});
