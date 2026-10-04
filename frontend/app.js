const MODEL_URL = "./DDON.onnx";
let rows = [], prediction = null, session = null, yTrue = null;
const statusEl = document.getElementById("status");

async function getSession() {
  if (session) return session;
  statusEl.textContent = "Loading DDON model (first use may take a moment)...";
  ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.23.2/dist/";
  session = await ort.InferenceSession.create(MODEL_URL, { executionProviders: ["wasm"] });
  return session;
}

function unwrap(x) {
  while (Array.isArray(x) && x.length === 1) x = x[0];
  return x;
}
function numericFlat(x) {
  x = unwrap(x);
  if (x == null) return [];
  if (ArrayBuffer.isView(x)) return Array.from(x, Number);
  if (typeof x === "number") return [x];
  if (Array.isArray(x)) return x.flat(Infinity).map(Number);
  if (x.data !== undefined) return numericFlat(x.data);
  if (x.real !== undefined) return numericFlat(x.real);
  return [];
}
function field(obj, name) {
  obj = unwrap(obj);
  if (obj && typeof obj === "object" && name in obj) return unwrap(obj[name]);
  return undefined;
}
function matrix109x2(x) {
  x = unwrap(x);
  if (Array.isArray(x) && x.length === 109 && Array.isArray(x[0])) {
    return x.map(r => [Number(r[0]), Number(r[1])]);
  }
  const flat = numericFlat(x);
  if (flat.length !== 218) throw new Error("Profile_Px.Px must contain 109 x 2 values.");
  // MATLAB stores matrices column-major: first 109 z, then 109 EFISH.
  return Array.from({length:109}, (_,i) => [flat[i], flat[i+109]]);
}
function scalar(x) {
  const a = numericFlat(x);
  if (!a.length || !Number.isFinite(a[0])) throw new Error("Profile_Px.u is missing or invalid.");
  return a[0];
}

async function loadMat(file) {
  const buffer = await file.arrayBuffer();
  let parsed;
  try {
    parsed = mat4js.read(buffer);
  } catch (e) {
    if ((e.feature || "").toUpperCase() === "HDF5" || /HDF5|7\.3/i.test(String(e))) {
      throw new Error("MATLAB v7.3/HDF5 is not supported yet. Re-save the file with MATLAB: save('file.mat', '-v7').");
    }
    throw e;
  }
  const data = parsed.data || parsed;
  const profile = field(data, "Profile_Px");
  if (!profile) throw new Error("MAT file does not contain Profile_Px.");

  rows = matrix109x2(field(profile, "Px"));
  const rawU = scalar(field(profile, "u"));
  document.getElementById("u").value = rawU;

  const ex = field(profile, "Ex");
  const exFlat = numericFlat(ex);
  yTrue = exFlat.length === 109 ? exFlat : null;

  statusEl.textContent =
    "MAT loaded: Profile_Px.Px (109 x 2), u=" + rawU +
    (yTrue ? ", Ex detected." : ".");
}

document.getElementById("file").addEventListener("change", async e => {
  const file = e.target.files[0];
  if (!file) return;
  yTrue = null;
  try {
    if (/\.mat$/i.test(file.name)) {
      await loadMat(file);
    } else {
      const text = await file.text();
      rows = text.split(/\r?\n/).filter(Boolean).map(line =>
        line.trim().split(/[\s,;]+/).map(Number)
      );
      statusEl.textContent = rows.length + " CSV rows loaded.";
    }
  } catch (err) {
    console.error(err);
    rows = [];
    statusEl.textContent = "Error reading file: " + err.message;
  }
});

document.getElementById("run").addEventListener("click", async () => {
  if (rows.length !== 109 || rows.some(r => r.length < 2 || !Number.isFinite(r[0]) || !Number.isFinite(r[1]))) {
    statusEl.textContent = "Input must contain exactly 109 valid points with z and EFISH.";
    return;
  }
  try {
    const s = await getSession();
    statusEl.textContent = "Running DDON in your browser...";
    const flat = new Float32Array(218);
    rows.forEach((r,i) => { flat[i*2]=r[0]; flat[i*2+1]=r[1]; });
    const rawU = Number(document.getElementById("u").value);
    if (!Number.isFinite(rawU)) throw new Error("u must be a valid number.");
    const valuesTensor = new ort.Tensor("float32", flat, [1,109,2]);
    const uTensor = new ort.Tensor("float32", new Float32Array([rawU / -1.0]), [1,1]);
    const feeds = {};
    feeds[s.inputNames[0]] = valuesTensor;
    feeds[s.inputNames[1]] = uTensor;
    const results = await s.run(feeds);
    const efield = Array.from(results[s.outputNames[0]].data);
    prediction = {model:"ddon", z:rows.map(r=>r[0]), efield};
    draw(prediction.z, prediction.efield, yTrue, rows.map(r => r[1]));
    document.getElementById("result").hidden=false;
    statusEl.textContent="Prediction complete (computed locally in your browser)." + (yTrue ? " Ex shown for comparison." : "");
  } catch(err) {
    console.error(err);
    statusEl.textContent="Error: "+err.message;
  }
});

function drawSeries(ctx,x,y,p,w,h,xmin,xmax,ymin,ymax,stroke) {
  const dy=(ymax-ymin)||1;
  ctx.strokeStyle=stroke; ctx.lineWidth=2; ctx.beginPath();
  y.forEach((v,i)=>{
    const px=p+(x[i]-xmin)/(xmax-xmin||1)*w, py=p+h-(v-ymin)/dy*h;
    i ? ctx.lineTo(px,py) : ctx.moveTo(px,py);
  });
  ctx.stroke();
}

function niceNumber(v) {
  if (Math.abs(v) >= 100 || (Math.abs(v) > 0 && Math.abs(v) < 0.01)) return v.toExponential(1);
  return Number(v.toFixed(2)).toString();
}

function drawAxes(ctx,p,w,h,xmin,xmax,ymin,ymax) {
  const ticks=5;
  ctx.strokeStyle="#222"; ctx.fillStyle="#222"; ctx.lineWidth=1; ctx.font="12px sans-serif";
  ctx.beginPath(); ctx.moveTo(p,p); ctx.lineTo(p,p+h); ctx.lineTo(p+w,p+h); ctx.stroke();

  ctx.textAlign="center"; ctx.textBaseline="top";
  for(let i=0;i<=ticks;i++){
    const t=i/ticks, px=p+t*w, value=xmin+t*(xmax-xmin);
    ctx.beginPath(); ctx.moveTo(px,p+h); ctx.lineTo(px,p+h+5); ctx.stroke();
    ctx.fillText(niceNumber(value),px,p+h+8);
  }

  ctx.textAlign="right"; ctx.textBaseline="middle";
  for(let i=0;i<=ticks;i++){
    const t=i/ticks, py=p+h-t*h, value=ymin+t*(ymax-ymin);
    ctx.beginPath(); ctx.moveTo(p-5,py); ctx.lineTo(p,py); ctx.stroke();
    ctx.fillText(niceNumber(value),p-8,py);
  }
}

function draw(x, y, trueY, inputP) {
  const c=document.getElementById("plot"),ctx=c.getContext("2d");
  const left=65,right=25,top=45,bottom=55;
  const p=left,w=c.width-left-right,h=c.height-top-bottom, yTop=top;

  ctx.clearRect(0,0,c.width,c.height);

  const allY=y.concat(inputP||[]).concat(trueY||[]);
  let xmin=Math.min(...x),xmax=Math.max(...x),ymin=Math.min(...allY),ymax=Math.max(...allY);
  if(ymin===ymax){ymin-=0.5;ymax+=0.5;}

  drawAxes(ctx,left,w,h,xmin,xmax,ymin,ymax);

  if(inputP) drawSeries(ctx,x,inputP,left,w,h,xmin,xmax,ymin,ymax,"#222222");
  drawSeries(ctx,x,y,left,w,h,xmin,xmax,ymin,ymax,"#d62728");
  if(trueY) drawSeries(ctx,x,trueY,left,w,h,xmin,xmax,ymin,ymax,"#1769aa");

  ctx.fillStyle="#222"; ctx.font="14px sans-serif"; ctx.textAlign="center"; ctx.textBaseline="alphabetic";
  ctx.fillText("Normalized z",left+w/2,c.height-8);
  ctx.save(); ctx.translate(16,yTop+h/2); ctx.rotate(-Math.PI/2);
  ctx.fillText("Normalized P / E",0,0); ctx.restore();

  ctx.font="14px sans-serif"; ctx.textAlign="left";
  let lx=left+10;
  if(inputP){ctx.fillStyle="#222222";ctx.fillText("Input P",lx,20);lx+=90;}
  ctx.fillStyle="#d62728";ctx.fillText("DDON prediction E",lx,20);lx+=145;
  if(trueY){ctx.fillStyle="#1769aa";ctx.fillText("Benchmark Ex",lx,20);}
}
document.getElementById("download").addEventListener("click",()=>{
  if(!prediction)return;
  const header=yTrue ? "z,efield,Ex" : "z,efield";
  const lines=[header,...prediction.z.map((z,i)=>yTrue ? z+","+prediction.efield[i]+","+yTrue[i] : z+","+prediction.efield[i])];
  const a=document.createElement("a");a.href=URL.createObjectURL(new Blob([lines.join("\n")],{type:"text/csv"}));
  a.download="ddon_prediction.csv";a.click();URL.revokeObjectURL(a.href);
});