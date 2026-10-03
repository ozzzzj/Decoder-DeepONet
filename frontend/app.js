let rows=[], prediction=null;
const statusEl=document.getElementById("status");
document.getElementById("file").addEventListener("change", async e=>{
  const text=await e.target.files[0].text();
  rows=text.split(/\r?\n/).filter(Boolean).map(line=>line.trim().split(/[\s,;]+/).map(Number));
  statusEl.textContent=rows.length+" rows loaded.";
});
document.getElementById("run").addEventListener("click", async ()=>{
  if(rows.length!==109 || rows.some(r=>r.length<2 || !Number.isFinite(r[0]) || !Number.isFinite(r[1]))){
    statusEl.textContent="Input must contain exactly 109 valid rows with two columns."; return;
  }
  statusEl.textContent="Running DDON...";
  try{
    const res=await fetch("/predict",{method:"POST",headers:{"Content-Type":"application/json"},
      body:JSON.stringify({model:"ddon",values:rows.map(r=>[r[0],r[1]]),u:Number(document.getElementById("u").value)})});
    const data=await res.json();
    if(!res.ok) throw new Error(data.detail||"Prediction failed");
    prediction=data; draw(data.z,data.efield);
    document.getElementById("result").hidden=false;
    statusEl.textContent="Prediction complete.";
  }catch(err){statusEl.textContent=err.message;}
});
function draw(x,y){
  const c=document.getElementById("plot"),ctx=c.getContext("2d"),p=45,w=c.width-2*p,h=c.height-2*p;
  ctx.clearRect(0,0,c.width,c.height); ctx.strokeStyle="#222"; ctx.lineWidth=1;
  ctx.beginPath();ctx.moveTo(p,p);ctx.lineTo(p,p+h);ctx.lineTo(p+w,p+h);ctx.stroke();
  const xmin=Math.min(...x),xmax=Math.max(...x),ymin=Math.min(...y),ymax=Math.max(...y),dy=(ymax-ymin)||1;
  ctx.strokeStyle="#1769aa";ctx.lineWidth=2;ctx.beginPath();
  y.forEach((v,i)=>{const px=p+(x[i]-xmin)/(xmax-xmin||1)*w,py=p+h-(v-ymin)/dy*h;i?ctx.lineTo(px,py):ctx.moveTo(px,py);});ctx.stroke();
  ctx.fillStyle="#222";ctx.font="14px sans-serif";ctx.fillText("Normalized z",c.width/2-35,c.height-8);
  ctx.save();ctx.translate(15,c.height/2+40);ctx.rotate(-Math.PI/2);ctx.fillText("Predicted E-field",0,0);ctx.restore();
}
document.getElementById("download").addEventListener("click",()=>{
  if(!prediction)return;
  const lines=["z,efield",...prediction.z.map((z,i)=>z+","+prediction.efield[i])];
  const a=document.createElement("a");a.href=URL.createObjectURL(new Blob([lines.join("\n")],{type:"text/csv"}));
  a.download="ddon_prediction.csv";a.click();URL.revokeObjectURL(a.href);
});