"""Self-contained joint-space preview; no server, CDNs or extra dependencies."""

# ruff: noqa: RUF001

import json

import numpy as np


def write_preview(plan, output):
    arrays = plan.arrays()
    payload = {"report": plan.report, "raw": plan.raw.tolist()}
    for phase in ("approach", "playback"):
        # Bound browser data size; analytic extrema are retained in report.
        indices = np.unique(np.linspace(0, len(arrays[phase + "/time"]) - 1, 600).astype(int))
        payload[phase] = {
            key: arrays[phase + "/" + key][indices].tolist() for key in ("time", "position", "velocity", "acceleration")
        }
    encoded = json.dumps(payload, allow_nan=False).replace("<", "\\u003c")
    html = """<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>权重动作评估 · 离线预览</title>
<style>
body{font:16px system-ui;margin:24px auto;padding:0 20px;max-width:1100px;color:#203044;background:#f5f7fa}
canvas{width:100%;height:290px;background:white;border:1px solid #d5dce5;border-radius:8px;margin:8px 0}
select,button{font:inherit;padding:6px;margin:4px} p{line-height:1.7}.note{color:#805015}
#cursor{width:70%} #status{font-variant-numeric:tabular-nums} small{color:#536578}
</style><h1>权重动作评估 · 离线预览</h1>
<p class="note">没有发送硬件命令。此图只显示关节角，不是机械臂三维路径，也没有完成碰撞检查。</p>
<p id="summary"></p><label>关节 <select id="joint"></select></label>
<label>指标 <select id="metric"><option value="position">位置 rad</option><option value="velocity">速度 rad/s</option>
<option value="acceleration">加速度 rad/s²</option></select></label>
<p><span style="color:#b77410">橙色：规划的首步接近</span>　<span style="color:#126aab">蓝色：平滑慢放的模型动作</span>
　<span style="color:#cf4366">红点：原始预测位置（仅拉伸时间对齐）</span></p>
<canvas id="chart" width="1100" height="320"></canvas>
<button id="play">播放时间指示</button><input id="cursor" type="range" min="0" max="1000" value="0"><p id="status"></p>
<small>切换关节检查双臂各 7 维与双手各 20 维。位置/速度/加速度连接处均规划为静止边界。
为便于浏览曲线做了抽样；是否超限以 report.json 的连续曲线极值检查为准。停稳等待是名义时间，真机必须检查实际反馈。</small>
<script id="data" type="application/json">__DATA__</script>
<script>
const D=JSON.parse(document.getElementById('data').textContent),R=D.report;
const $=id=>document.getElementById(id),canvas=$('chart'),ctx=canvas.getContext('2d');
R.joint_names.forEach((name,i)=>{let o=document.createElement('option');o.value=i;o.textContent=name;$('joint').appendChild(o)});
$('summary').textContent=`首步接近 ${R.approach_seconds.toFixed(2)} s；模型播放 ${R.playback_seconds.toFixed(2)} s；实际时间倍率 ${R.playback_time_scale.toFixed(2)}；名义总时长 ${R.total_seconds.toFixed(2)} s。`;
let playing=false,last=null;
function draw(){let j=+$('joint').value,m=$('metric').value,off=R.approach_seconds+R.settle_seconds;
let ap=D.approach[m].map(x=>x[j]),pb=D.playback[m].map(x=>x[j]);let raw=D.raw.map(x=>x[j]);
let values=ap.concat(pb,m==='position'?raw:[0]),lo=Math.min(...values),hi=Math.max(...values),pad=Math.max((hi-lo)*.1,.001);lo-=pad;hi+=pad;
const X=t=>70+t/R.total_seconds*1000,Y=q=>270-(q-lo)/(hi-lo)*230;
ctx.clearRect(0,0,1100,320);ctx.font='13px system-ui';ctx.lineWidth=1;
for(let i=0;i<=5;i++){let q=lo+(hi-lo)*i/5;ctx.strokeStyle='#e4e9ef';ctx.beginPath();ctx.moveTo(70,Y(q));ctx.lineTo(1070,Y(q));ctx.stroke();ctx.fillStyle='#536578';ctx.fillText(q.toFixed(3),8,Y(q)+4)}
for(let i=0;i<=5;i++){let t=R.total_seconds*i/5;ctx.fillText(t.toFixed(1)+'s',X(t)-10,298)}
function line(ts,qs,offset,color){ctx.strokeStyle=color;ctx.lineWidth=2;ctx.beginPath();ts.forEach((t,i)=>{if(i)ctx.lineTo(X(t+offset),Y(qs[i]));else ctx.moveTo(X(t+offset),Y(qs[i]))});ctx.stroke()}
line(D.approach.time,ap,0,'#b77410');line(D.playback.time,pb,off,'#126aab');
line([R.approach_seconds,off],[ap.at(-1),pb[0]],0,'#8b97a5');
if(m==='position'){ctx.fillStyle='#cf4366';raw.forEach((q,i)=>{ctx.beginPath();ctx.arc(X(off+i/30*R.playback_time_scale),Y(q),2.5,0,7);ctx.fill()})}
let t=+$('cursor').value/1000*R.total_seconds;ctx.strokeStyle='#212936';ctx.beginPath();ctx.moveTo(X(t),35);ctx.lineTo(X(t),275);ctx.stroke();
let phase=t<R.approach_seconds?'首步接近':t<off?'等待到位':t<off+R.playback_seconds?'模型播放':'等待停止';
$('status').textContent=`${t.toFixed(2)} s · ${phase} · ${R.joint_names[j]}`;
}
$('joint').onchange=$('metric').onchange=$('cursor').oninput=draw;
$('play').onclick=()=>{playing=!playing;last=null;$('play').textContent=playing?'暂停时间指示':'播放时间指示'};
function frame(stamp){if(playing){if(last!==null){let n=+$('cursor').value+(stamp-last)/1000/R.total_seconds*1000;$('cursor').value=n>=1000?0:n;draw()}last=stamp}requestAnimationFrame(frame)}draw();requestAnimationFrame(frame);
</script></html>"""
    output.write_text(html.replace("__DATA__", encoded))
