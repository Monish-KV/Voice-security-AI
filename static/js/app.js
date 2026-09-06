const $ = id => document.getElementById(id);
let selectedFile = null;
let referenceFile = null;
let recordedBlob = null;
let mediaRecorder = null;
let mediaStream = null;
let chunks = [];
let timerHandle = null;
let recordStartedAt = 0;
let currentResult = null;

function fmt(sec) { sec = Math.max(0, Math.floor(Number(sec) || 0)); return `${Math.floor(sec/60)}:${String(sec%60).padStart(2,'0')}`; }
function pct(v) { const n = Number(v); return Number.isFinite(n) ? `${n.toFixed(1)}%` : '—'; }
function setBar(id, value) { $(id).style.width = `${Math.max(0, Math.min(100, Number(value)||0))}%`; }

// Navigation
 document.querySelectorAll('.nav-item').forEach(btn => btn.addEventListener('click', () => {
   document.querySelectorAll('.nav-item').forEach(x=>x.classList.remove('active')); btn.classList.add('active');
   const target = btn.dataset.target;

   // Live Monitor is a mode inside Detection, not a separate page section.
   if (target === 'live') {
     document.querySelectorAll('.page-section').forEach(s=>s.classList.remove('active-section'));
     $('detection').classList.add('active-section');
     document.querySelectorAll('.mode-tab').forEach(x=>x.classList.remove('active'));
     document.querySelector('[data-mode="live"]').classList.add('active');
     $('recordedPanel').classList.remove('active-mode');
     $('livePanel').classList.add('active-mode');
     $('detection').scrollIntoView({behavior:'smooth', block:'start'});
     return;
   }

   document.querySelectorAll('.page-section').forEach(s=>s.classList.remove('active-section'));
   const section = $(target);
   if (!section) return;
   section.classList.add('active-section');

   if (target === 'detection') {
     document.querySelectorAll('.mode-tab').forEach(x=>x.classList.remove('active'));
     document.querySelector('[data-mode="recorded"]').classList.add('active');
     $('recordedPanel').classList.add('active-mode');
     $('livePanel').classList.remove('active-mode');
   }
   if (target === 'history') loadHistory();
   if (target === 'privacy') loadPrivacy();
   if (target === 'settings') loadSettings();
 }));

// Mode tabs
 document.querySelectorAll('.mode-tab').forEach(btn => btn.addEventListener('click', () => {
   document.querySelectorAll('.mode-tab').forEach(x=>x.classList.remove('active')); btn.classList.add('active');
   const live = btn.dataset.mode === 'live'; $('recordedPanel').classList.toggle('active-mode', !live); $('livePanel').classList.toggle('active-mode', live);
 }));

$('chooseBtn').onclick = () => $('audioFile').click();
$('referenceBtn').onclick = () => $('referenceFile').click();
$('audioFile').onchange = e => { selectedFile = e.target.files[0] || null; $('fileName').textContent = selectedFile ? selectedFile.name : 'No recording selected'; $('analyzeRecorded').disabled = !selectedFile; };
$('referenceFile').onchange = e => { referenceFile = e.target.files[0] || null; $('referenceName').textContent = referenceFile ? referenceFile.name : 'Optional'; };

async function startRecording() {
  try {
    mediaStream = await navigator.mediaDevices.getUserMedia({audio:true});
    chunks = []; mediaRecorder = new MediaRecorder(mediaStream); recordStartedAt = Date.now();
    mediaRecorder.ondataavailable = e => { if(e.data.size) chunks.push(e.data); };
    mediaRecorder.onstop = () => {
      recordedBlob = new Blob(chunks, {type: mediaRecorder.mimeType || 'audio/webm'});
      selectedFile = new File([recordedBlob], 'microphone_recording.webm', {type: recordedBlob.type});
      $('recordStatus').textContent = `Recording ready · ${fmt((Date.now()-recordStartedAt)/1000)}`;
      $('analyzeLive').disabled = false; mediaStream?.getTracks().forEach(t=>t.stop()); mediaStream=null;
    };
    mediaRecorder.start(250); $('startBtn').disabled=true; $('stopBtn').disabled=false; $('recordOrb').classList.add('recording');
    $('recordStatus').textContent='Recording in progress…'; timerHandle=setInterval(()=>{$('recordTimer').textContent=fmt((Date.now()-recordStartedAt)/1000)},250);
  } catch(e) { $('recordStatus').textContent='Microphone unavailable or permission denied.'; }
}
function stopRecording(){ if(!mediaRecorder) return; mediaRecorder.stop(); $('startBtn').disabled=false; $('stopBtn').disabled=true; $('recordOrb').classList.remove('recording'); clearInterval(timerHandle); }
$('startBtn').onclick=startRecording; $('stopBtn').onclick=stopRecording;

function buildForm(sourceType) {
  const fd = new FormData(); fd.append('audio', selectedFile); fd.append('source_type', sourceType);
  const live = sourceType === 'live';
  fd.append('language_profile', $(live?'liveLanguage':'languageProfile').value);
  fd.append('contact_context', $(live?'liveContact':'contactContext').value);
  fd.append('transaction_context', $(live?'liveTransaction':'transactionContext').value);
  fd.append('unknown_caller', String($(live?'liveUnknown':'unknownCaller').checked));
  fd.append('sensitive_transaction', String($(live?'liveSensitive':'sensitiveTransaction').checked));
  fd.append('first_contact', String($(live?'liveFirst':'firstContact').checked));
  if(!live && referenceFile) fd.append('reference_audio', referenceFile);
  return fd;
}
async function analyze(sourceType) {
  const btn = sourceType==='live' ? $('analyzeLive') : $('analyzeRecorded'); if(!selectedFile) return;
  btn.disabled=true; const old=btn.textContent; btn.textContent='ANALYZING…';
  try { const r=await fetch('/predict',{method:'POST',body:buildForm(sourceType)}); const data=await r.json(); if(!r.ok||!data.success) throw new Error(data.error||'Analysis failed'); currentResult=data; displayResults(data,sourceType); }
  catch(e){ alert(`Analysis failed: ${e.message}`); }
  finally { btn.disabled=false; btn.textContent=old; }
}
$('analyzeRecorded').onclick=()=>analyze('recorded'); $('analyzeLive').onclick=()=>analyze('live');

function displayResults(data, sourceType){
  const r=data.result; $('results').classList.add('active-section'); $('results').classList.remove('hidden'); $('resultTitle').textContent=sourceType==='live'?'Live Voice Analysis':'Recorded Voice Analysis'; $('resultMeta').textContent=`${r.duration}s · ${sourceType==='live'?'microphone session':'uploaded recording'} · ${r.thresholds.medium}% / ${r.thresholds.high}% thresholds`;
  $('riskScore').textContent=pct(r.risk_score); $('riskBadge').textContent=r.risk_level; $('classification').textContent=r.risk_level==='HIGH'?'Potential Voice Impersonation':r.risk_level==='MEDIUM'?'Suspicious Voice Characteristics':'Likely Genuine Voice';
  $('riskDescription').textContent=r.risk_level==='HIGH'?'Strong synthetic-risk evidence. Sensitive authorization should be blocked pending independent verification.':r.risk_level==='MEDIUM'?'Some synthetic-risk evidence was detected. Use additional identity verification.':'No strong synthetic characteristics detected by the current models.';
  $('contextAdjustment').textContent=`Context +${Number(r.context_adjustment).toFixed(1)}`;
  $('v2Score').textContent=`${pct(r.v2_spoof_probability)} SPOOF`; $('v4Score').textContent=`${pct(r.v4_spoof_probability)} SPOOF`; renderV5Secondary(r); $('modelRisk').textContent=pct(r.model_risk_score); $('finalScore').textContent=pct(r.risk_score);
  setBar('v2Bar',r.v2_spoof_probability); setBar('v4Bar',r.v4_spoof_probability); setBar('modelBar',r.model_risk_score); setBar('finalBar',r.risk_score);
  $('recommendation').textContent=r.recommendation; $('auditId').textContent=data.audit_event_id;
  const requirements=[]; if(r.response_requirements?.callback) requirements.push('Independent callback required'); if(r.response_requirements?.mfa) requirements.push('MFA required'); if(!requirements.length) requirements.push('Standard verification policy applies'); $('responseRequirements').textContent=requirements.join(' · ');
  $('pitchVariation').textContent=r.prosody ? r.prosody.pitch_variation : '—'; $('energyVariation').textContent=r.prosody ? r.prosody.energy_variation : '—'; $('pauseRatio').textContent=r.prosody ? pct(r.prosody.pause_ratio) : '—'; $('speakingActivity').textContent=r.prosody ? pct(r.prosody.speaking_activity) : '—';
  if(r.speaker_consistency_available){ $('consistencyScore').textContent=pct(r.speaker_consistency_score); $('consistencyText').textContent='Higher values indicate greater acoustic similarity to the supplied genuine reference. This is a supporting consistency signal, not biometric identification.'; } else { $('consistencyScore').textContent='Not supplied'; $('consistencyText').textContent='Add a known genuine reference recording to compare compact acoustic signatures across sessions.'; }
  renderTimeline(r); $('analysisNote').textContent=r.analysis_note; setRiskStyle(r.risk_level); document.querySelectorAll('.page-section').forEach(s=>s.classList.remove('active-section')); $('results').classList.add('active-section'); $('results').scrollIntoView({behavior:'smooth'});
}
function setRiskStyle(level){ $('riskBadge').className=`risk-badge ${level.toLowerCase()}`; $('riskScore').className=level.toLowerCase(); }
function renderTimeline(r){ const el=$('timeline'); el.innerHTML=''; const total=Number(r.duration)||0; $('durationLabel').textContent=fmt(total); $('timelineEnd').textContent=fmt(total); if(!r.timeline?.length){el.innerHTML='<div class="timeline-empty">No segment data</div>';return;} r.timeline.forEach(seg=>{const d=document.createElement('div'); const width=Math.max(0.5,((seg.end-seg.start)/total)*100); d.className=`timeline-segment ${seg.label.toLowerCase()}`; d.style.width=`${width}%`; d.title=`${fmt(seg.start)} – ${fmt(seg.end)} · ${seg.label}`; d.innerHTML=`<b>${seg.label}</b><small>${fmt(seg.start)}–${fmt(seg.end)}</small>`; el.appendChild(d);}); }

async function recordSecurityAction(action, riskLevel, sourceEventId=''){
  const response=await fetch('/api/security-action',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({action,risk_level:riskLevel,source_event_id:sourceEventId})});
  const data=await response.json();
  if(!response.ok||!data.success) throw new Error(data.error||'Could not record the security action');
  return data.event;
}
$('workflowBtn').onclick=async()=>{ if(!currentResult) return; const btn=$('workflowBtn'); const original=btn.textContent; btn.disabled=true; try { await recordSecurityAction(currentResult.result.recommended_action,currentResult.result.risk_level,currentResult.audit_event_id); btn.textContent='Action Recorded'; } catch(e) { alert(e.message); } finally { setTimeout(()=>{btn.disabled=false;btn.textContent=original;},1600); } };
document.querySelectorAll('.security-action').forEach(btn=>btn.onclick=async()=>{ const original=btn.textContent; btn.disabled=true; try { const event=await recordSecurityAction(btn.dataset.action,btn.dataset.risk,currentResult?.audit_event_id||''); $('securityActionStatus').textContent=`✓ ${event.action} recorded at ${new Date(event.timestamp).toLocaleTimeString()}.`; } catch(e) { $('securityActionStatus').textContent=`✕ ${e.message}`; } finally { setTimeout(()=>{btn.disabled=false;btn.textContent=original;},900); } });

async function loadHistory(){ const r=await fetch('/api/events'); const d=await r.json(); const body=$('historyBody'); body.innerHTML=''; (d.events||[]).forEach(e=>{const tr=document.createElement('tr'); tr.innerHTML=`<td>${new Date(e.timestamp).toLocaleString()}</td><td>${e.source_type||e.event_type||'—'}</td><td><span class="table-risk ${String(e.risk_level||'').toLowerCase()}">${e.risk_level||'—'}</span></td><td>${e.risk_score??'—'}</td><td>${e.raw_audio_retained===false?'NO':'—'}</td><td><code>${e.event_id||'—'}</code></td>`;body.appendChild(tr);}); if(!body.children.length) body.innerHTML='<tr><td colspan="6">No events yet.</td></tr>'; }
$('verifyAudit').onclick=async()=>{const d=await (await fetch('/api/audit/verify')).json();$('historyStatus').textContent=`${d.valid?'✓':'✕'} ${d.message} Checked: ${d.checked}`;$('historyStatus').className=`verification ${d.valid?'valid':'invalid'}`;};
async function loadPrivacy(){
  try { const response=await fetch('/api/privacy'); const d=await response.json(); if(!response.ok||!d.success) throw new Error('Privacy status unavailable');
    $('rawAudioStatus').textContent=d.raw_audio_retained?'ON':'OFF';
    $('referenceAudioStatus').textContent=d.reference_audio_retained?'RETAINED':'EPHEMERAL';
    $('auditStorageStatus').textContent=`${d.audit_event_count} EVENT${d.audit_event_count===1?'':'S'}`;
    $('auditStorageDetail').textContent=d.last_event_at?`Latest metadata event: ${new Date(d.last_event_at).toLocaleString()}. Raw voice is never retained.`:'No metadata events have been recorded yet.';
    $('privacyIntegrityStatus').textContent=d.audit_chain_valid?'VERIFIED':'CHECK REQUIRED';
    $('privacyIntegrityDetail').textContent=d.audit_chain_valid?`Hash chain verified across ${d.checked_events} chained event${d.checked_events===1?'':'s'}.`:'The audit hash chain needs review in Event History.';
  } catch(e) { $('privacyIntegrityStatus').textContent='UNAVAILABLE'; $('privacyIntegrityDetail').textContent=e.message; }
}
$('refreshPrivacy').onclick=loadPrivacy;
async function loadSettings(){const d=await (await fetch('/api/settings')).json();const s=d.settings;$('mediumThreshold').value=s.medium_threshold;$('highThreshold').value=s.high_threshold;$('mfaHigh').checked=s.mfa_on_high;$('callbackHigh').checked=s.callback_on_high;$('protectSensitive').checked=s.protect_sensitive_transactions;}
$('saveSettings').onclick=async()=>{const payload={medium_threshold:Number($('mediumThreshold').value),high_threshold:Number($('highThreshold').value),mfa_on_high:$('mfaHigh').checked,callback_on_high:$('callbackHigh').checked,protect_sensitive_transactions:$('protectSensitive').checked};try{const r=await fetch('/api/settings',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)});const d=await r.json();if(!r.ok||!d.success)throw new Error(d.error||'Save failed');$('mediumThreshold').value=d.settings.medium_threshold;$('highThreshold').value=d.settings.high_threshold;$('settingsStatus').textContent='✓ Settings saved and will be used by new analyses';}catch(e){$('settingsStatus').textContent=`✕ ${e.message}`;}};

function renderV5Secondary(r){
  const el=document.getElementById('v5SecondaryScore'), box=document.getElementById('v5SecondaryCard');
  if(!el||!box)return;
  if(r.v5_secondary_spoof_probability===null||r.v5_secondary_spoof_probability===undefined){el.textContent='Unavailable';return;}
  el.textContent=pct(r.v5_secondary_spoof_probability)+' SPOOF';
  box.classList.toggle('disagreement',!!r.v5_disagreement);
  const note=box.querySelector('.v5-note');
  if(note) note.textContent=r.v5_disagreement?'⚠️ V2+V4 and Raw V5 disagree — additional verification recommended.':'Secondary check agrees with the primary assessment.';
}
