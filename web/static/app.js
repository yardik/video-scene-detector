/**
 * Video Scene Detector — Web Client JavaScript Application
 */

document.addEventListener('DOMContentLoaded', () => {
  // DOM Element References
  const inputDirInput = document.getElementById('inputDir');
  const outputDirInput = document.getElementById('outputDir');
  const checkInputBtn = document.getElementById('checkInputBtn');
  const inputFolderInfo = document.getElementById('inputFolderInfo');
  
  const promptInput = document.getElementById('promptInput');
  const qualityBadge = document.getElementById('qualityBadge');
  const conditionsPreview = document.getElementById('conditionsPreview');
  const analyzerHint = document.getElementById('analyzerHint');
  
  const modelSelect = document.getElementById('modelSelect');
  const quantSelect = document.getElementById('quantSelect');
  
  const startBatchBtn = document.getElementById('startBatchBtn');
  const stopBatchBtn = document.getElementById('stopBatchBtn');
  
  const vramBadge = document.getElementById('vramBadge');
  const statusBadge = document.getElementById('statusBadge');
  
  const currentVideoLabel = document.getElementById('currentVideoLabel');
  const progressPercentLabel = document.getElementById('progressPercentLabel');
  const progressBar = document.getElementById('progressBar');
  const phaseLabel = document.getElementById('phaseLabel');
  const scenesFoundLabel = document.getElementById('scenesFoundLabel');
  const elapsedLabel = document.getElementById('elapsedLabel');
  
  const logOutput = document.getElementById('logOutput');
  const clearLogsBtn = document.getElementById('clearLogsBtn');
  
  const refreshClipsBtn = document.getElementById('refreshClipsBtn');
  const clipGrid = document.getElementById('clipGrid');
  const clipCountBadge = document.getElementById('clipCountBadge');
  const galleryPathHint = document.getElementById('galleryPathHint');
  
  const videoModal = document.getElementById('videoModal');
  const modalTitle = document.getElementById('modalTitle');
  const modalVideoPlayer = document.getElementById('modalVideoPlayer');
  const closeModalBtn = document.getElementById('closeModalBtn');

  // EventSource for SSE Progress Streaming
  let eventSource = null;
  let promptDebounceTimer = null;

  // System Prompts Studio DOM Elements
  const togglePromptsBtn = document.getElementById('togglePromptsBtn');
  const promptsToggleIcon = document.getElementById('promptsToggleIcon');
  const promptsBody = document.getElementById('promptsBody');
  const captionerPromptInput = document.getElementById('captionerPromptInput');
  const matcherPromptInput = document.getElementById('matcherPromptInput');
  const savePromptsBtn = document.getElementById('savePromptsBtn');
  const resetPromptsBtn = document.getElementById('resetPromptsBtn');
  const promptsStatusMsg = document.getElementById('promptsStatusMsg');

  // Initializations
  analyzePrompt();
  checkFolder();
  loadClips();
  loadSystemPrompts();

  // Toggle Prompts Studio Accordion
  if (togglePromptsBtn) {
    togglePromptsBtn.addEventListener('click', () => {
      promptsBody.classList.toggle('hidden');
      const isHidden = promptsBody.classList.contains('hidden');
      promptsToggleIcon.textContent = isHidden ? '▶' : '▼';
    });
  }

  // Load System Prompts on Startup
  async function loadSystemPrompts() {
    try {
      const res = await fetch('/api/prompts');
      const data = await res.json();
      if (data.captioner_prompt) captionerPromptInput.value = data.captioner_prompt;
      if (data.matcher_prompt) matcherPromptInput.value = data.matcher_prompt;
    } catch (err) {
      console.error('Failed to load system prompts:', err);
    }
  }

  // Save System Prompts to Disk
  async function saveSystemPrompts() {
    promptsStatusMsg.textContent = 'Saving...';
    promptsStatusMsg.style.color = 'var(--text-muted)';
    try {
      const res = await fetch('/api/prompts', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          captioner_prompt: captionerPromptInput.value.trim(),
          matcher_prompt: matcherPromptInput.value.trim(),
        }),
      });
      const data = await res.json();
      if (data.status === 'success') {
        promptsStatusMsg.textContent = '✓ Saved to disk!';
        promptsStatusMsg.style.color = '#10b981';
        setTimeout(() => { promptsStatusMsg.textContent = ''; }, 3000);
      }
    } catch (err) {
      promptsStatusMsg.textContent = '✗ Error saving prompts';
      promptsStatusMsg.style.color = '#ef4444';
    }
  }

  // Reset System Prompts to Defaults
  async function resetSystemPrompts() {
    if (!confirm('Reset all system prompts to default templates?')) return;
    promptsStatusMsg.textContent = 'Resetting...';
    try {
      const res = await fetch('/api/prompts', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ reset_defaults: true }),
      });
      const data = await res.json();
      if (data.captioner_prompt) captionerPromptInput.value = data.captioner_prompt;
      if (data.matcher_prompt) matcherPromptInput.value = data.matcher_prompt;
      promptsStatusMsg.textContent = '✓ Reset to defaults!';
      promptsStatusMsg.style.color = '#10b981';
      setTimeout(() => { promptsStatusMsg.textContent = ''; }, 3000);
    } catch (err) {
      promptsStatusMsg.textContent = '✗ Error resetting prompts';
      promptsStatusMsg.style.color = '#ef4444';
    }
  }

  if (savePromptsBtn) savePromptsBtn.addEventListener('click', saveSystemPrompts);
  if (resetPromptsBtn) resetPromptsBtn.addEventListener('click', resetSystemPrompts);

  // ── Event Listeners ──

  promptInput.addEventListener('input', () => {
    clearTimeout(promptDebounceTimer);
    promptDebounceTimer = setTimeout(analyzePrompt, 400);
  });

  checkInputBtn.addEventListener('click', checkFolder);
  startBatchBtn.addEventListener('click', startBatch);
  stopBatchBtn.addEventListener('click', stopBatch);
  clearLogsBtn.addEventListener('click', () => { logOutput.textContent = ''; });
  refreshClipsBtn.addEventListener('click', loadClips);

  // Load installed models into all dropdowns on page load
  loadInstalledModels();

  closeModalBtn.addEventListener('click', closeModal);
  videoModal.querySelector('.modal-backdrop').addEventListener('click', closeModal);

  // Tab Switching Logic
  document.querySelectorAll('.tab-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
      document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
      
      btn.classList.add('active');
      const tabId = btn.getAttribute('data-tab');
      document.getElementById(tabId).classList.add('active');

      if (tabId === 'galleryTab') {
        loadClips();
      }
    });
  });

  // ── Functions ──

  async function analyzePrompt() {
    const prompt = promptInput.value.trim();
    if (!prompt) {
      qualityBadge.textContent = '--';
      qualityBadge.style.color = 'var(--text-muted)';
      conditionsPreview.innerHTML = '';
      analyzerHint.textContent = 'Enter a prompt to analyze quality';
      return;
    }

    try {
      const res = await fetch('/api/analyze-prompt', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ prompt }),
      });
      const data = await res.json();

      const pct = Math.round(data.score * 100);
      qualityBadge.textContent = `${pct}% (${pct >= 80 ? 'Excellent' : pct >= 60 ? 'Good' : 'Fair'})`;
      qualityBadge.style.color = pct >= 80 ? '#34d399' : pct >= 60 ? '#f59e0b' : '#ef4444';

      conditionsPreview.innerHTML = '';
      data.issues.forEach((issue) => {
        const tag = document.createElement('span');
        tag.className = 'cond-tag';
        tag.textContent = issue;
        conditionsPreview.appendChild(tag);
      });

      if (data.suggestions.length > 0) {
        analyzerHint.textContent = data.suggestions.join(' ');
      } else {
        analyzerHint.textContent = 'No issues detected.';
      }
    } catch (e) {
      console.error('Prompt analysis error:', e);
    }
  }

  async function checkFolder() {
    const path = inputDirInput.value.trim();
    if (!path) return;

    try {
      const res = await fetch('/api/check-path', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ path }),
      });
      const data = await res.json();

      if (data.exists) {
        if (data.is_file) {
          inputFolderInfo.textContent = `✓ Single video file target: ${data.video_names[0] || '1 video'}`;
          inputFolderInfo.style.color = '#34d399';
        } else if (data.is_dir) {
          inputFolderInfo.textContent = `✓ Found ${data.video_count} video file(s) in folder`;
          inputFolderInfo.style.color = '#34d399';
        } else {
          inputFolderInfo.textContent = `❌ Path is not a valid video file or directory`;
          inputFolderInfo.style.color = '#ef4444';
        }
      } else {
        inputFolderInfo.textContent = `❌ Path invalid or file/folder not found`;
        inputFolderInfo.style.color = '#ef4444';
      }
    } catch (e) {
      console.error('Check folder error:', e);
    }
  }

  const dualModelCheck = document.getElementById('dualModelCheck');
  const textModelGroup = document.getElementById('textModelGroup');
  const textModelSelect = document.getElementById('textModelSelect');
  const similarityThresholdSlider = document.getElementById('similarityThresholdSlider');
  const similarityThresholdValue = document.getElementById('similarityThresholdValue');
  const coarseWindowSlider = document.getElementById('coarseWindowSlider');
  const coarseWindowValue = document.getElementById('coarseWindowValue');
  const coarseFramesSlider = document.getElementById('coarseFramesSlider');
  const coarseFramesValue = document.getElementById('coarseFramesValue');
  const microBatchSlider = document.getElementById('microBatchSlider');
  const microBatchValue = document.getElementById('microBatchValue');
  const textLlmBatchSlider = document.getElementById('textLlmBatchSlider');
  const textLlmBatchValue = document.getElementById('textLlmBatchValue');
  const sageAttnCheck = document.getElementById('sageAttnCheck');

  if (dualModelCheck && textModelGroup) {
    dualModelCheck.addEventListener('change', () => {
      if (dualModelCheck.checked) {
        textModelGroup.classList.remove('hidden');
      } else {
        textModelGroup.classList.add('hidden');
      }
    });
  }

  if (similarityThresholdSlider && similarityThresholdValue) {
    similarityThresholdSlider.addEventListener('input', () => {
      similarityThresholdValue.textContent = `${similarityThresholdSlider.value}%`;
    });
  }

  function updateMicroBatchBounds(forceResetToMax = false) {
    if (!coarseFramesSlider || !microBatchSlider || !microBatchValue) return;
    const sampledFrames = parseInt(coarseFramesSlider.value) || 16;
    const oldMax = parseInt(microBatchSlider.max) || 64;
    const oldVal = parseInt(microBatchSlider.value) || 64;

    microBatchSlider.max = sampledFrames;

    if (forceResetToMax || oldVal >= oldMax || oldVal > sampledFrames) {
      microBatchSlider.value = sampledFrames;
    }
    microBatchValue.textContent = `${microBatchSlider.value} frames`;
  }

  if (coarseWindowSlider && coarseWindowValue) {
    coarseWindowSlider.addEventListener('input', () => {
      coarseWindowValue.textContent = `${parseFloat(coarseWindowSlider.value).toFixed(1)} min`;
    });
  }

  if (coarseFramesSlider && coarseFramesValue) {
    coarseFramesSlider.addEventListener('input', () => {
      coarseFramesValue.textContent = `${coarseFramesSlider.value} frames`;
      updateMicroBatchBounds();
    });
  }

  if (microBatchSlider && microBatchValue) {
    microBatchSlider.addEventListener('input', () => {
      const maxVal = parseInt(microBatchSlider.max) || 64;
      if (parseInt(microBatchSlider.value) > maxVal) {
        microBatchSlider.value = maxVal;
      }
      microBatchValue.textContent = `${microBatchSlider.value} frames`;
    });
  }

  if (textLlmBatchSlider && textLlmBatchValue) {
    textLlmBatchSlider.addEventListener('input', () => {
      textLlmBatchValue.textContent = `${textLlmBatchSlider.value} items`;
    });
  }

  updateMicroBatchBounds(true);

  async function startBatch() {
    const inputDir = inputDirInput.value.trim();
    const outputDir = outputDirInput.value.trim();
    const prompt = promptInput.value.trim();

    if (!inputDir || !prompt) {
      alert('Please fill out input video path and target prompt!');
      return;
    }

    startBatchBtn.classList.add('hidden');
    stopBatchBtn.classList.remove('hidden');
    statusBadge.textContent = 'Scanning...';
    statusBadge.style.color = '#6366f1';

    logOutput.textContent = `🚀 Initializing batch scan...\n`;

    // Connect SSE Stream
    connectSSE();

    try {
      const res = await fetch('/api/start-batch', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          input_dir: inputDir,
          output_dir: outputDir,
          prompt: prompt,
          model_name: modelSelect.value,
          quantization: quantSelect.value,
          dual_model: dualModelCheck ? dualModelCheck.checked : false,
          text_model_name: textModelSelect ? textModelSelect.value : null,
          similarity_threshold: similarityThresholdSlider ? parseFloat(similarityThresholdSlider.value) : 50.0,
          coarse_window_minutes: coarseWindowSlider ? parseFloat(coarseWindowSlider.value) : 3.0,
          coarse_frames: coarseFramesSlider ? parseInt(coarseFramesSlider.value) : 16,
          smart_motion_sampling: document.getElementById('smartMotionCheck') ? document.getElementById('smartMotionCheck').checked : false,
          use_sage_attention: sageAttnCheck ? sageAttnCheck.checked : true,
          image_only_micro_batch_size: microBatchSlider ? parseInt(microBatchSlider.value) : 64,
          text_llm_batch_size: textLlmBatchSlider ? parseInt(textLlmBatchSlider.value) : 10,
        }),
      });

      if (!res.ok) {
        const err = await res.json();
        alert(`Error starting batch: ${err.detail}`);
        resetButtons();
      }
    } catch (e) {
      console.error('Start batch error:', e);
      resetButtons();
    }
  }

  async function stopBatch() {
    try {
      appendLog('🛑 Stop signal sent to detector...');
      await fetch('/api/stop-batch', { method: 'POST' });
      resetButtons();
      statusBadge.textContent = 'Stopped';
      statusBadge.style.color = '#f59e0b';
    } catch (e) {
      console.error('Stop batch error:', e);
      resetButtons();
    }
  }

  function connectSSE() {
    if (eventSource) {
      eventSource.close();
    }

    eventSource = new EventSource('/api/progress-stream');

    eventSource.onmessage = (e) => {
      try {
        const data = JSON.parse(e.data);
        updateUIFromSSE(data);
      } catch (err) {
        console.error('SSE parse error:', err);
      }
    };

    eventSource.onerror = () => {
      console.log('SSE connection closed or lost.');
    };
  }

  function updateUIFromSSE(data) {
    if (data.log) {
      appendLog(data.log);

      // Parse step info from log lines if available (e.g. Window 5/26)
      const match = data.log.match(/Window\s+(\d+)\/(\d+)/i);
      if (match) {
        const curWin = parseInt(match[1]);
        const totWin = parseInt(match[2]);
        if (totWin > 0) {
          const scanPct = Math.round((curWin / totWin) * 100);
          const currentScanPercentLabel = document.getElementById('currentScanPercentLabel');
          const currentScanProgressBar = document.getElementById('currentScanProgressBar');
          if (currentScanPercentLabel && currentScanProgressBar) {
            currentScanPercentLabel.textContent = `${scanPct}% (Window ${curWin}/${totWin})`;
            currentScanProgressBar.style.width = `${scanPct}%`;
          }
        }
      }
    }

    if (data.vram && data.vram.active_mb) {
      vramBadge.textContent = `GPU VRAM: ${data.vram.active_mb} MB (${data.vram.reserved_mb} MB pool)`;
    }

    if (data.progress) {
      const p = data.progress;

      const batchVideoCountLabel = document.getElementById('batchVideoCountLabel');
      if (batchVideoCountLabel) {
        batchVideoCountLabel.textContent = `${p.current_video_idx} / ${p.total_videos} Videos`;
      }

      if (p.current_video_name) {
        currentVideoLabel.textContent = `🎬 [${p.current_video_idx}/${p.total_videos}] ${p.current_video_name}`;
      } else {
        currentVideoLabel.textContent = `Scanning videos... (${p.total_videos} total)`;
      }

      const pct = Math.round(p.percent_complete);
      progressPercentLabel.textContent = `${pct}%`;
      progressBar.style.width = `${pct}%`;

      phaseLabel.textContent = p.current_phase || '--';
      scenesFoundLabel.textContent = p.scenes_found_total || 0;
      elapsedLabel.textContent = `${p.elapsed_seconds || 0}s`;

      if (p.status === 'completed' || p.status === 'stopped' || p.status === 'error') {
        resetButtons();
        statusBadge.textContent = p.status === 'completed' ? 'Completed ✓' : 'Stopped';
        statusBadge.style.color = p.status === 'completed' ? '#34d399' : '#f59e0b';
        loadClips();
      }
    }
  }

  function appendLog(line) {
    logOutput.textContent += `${line}\n`;
    logOutput.scrollTop = logOutput.scrollHeight;
  }

  function resetButtons() {
    startBatchBtn.classList.remove('hidden');
    stopBatchBtn.classList.add('hidden');
  }

  async function loadClips() {
    const outputDir = outputDirInput.value.trim();
    galleryPathHint.textContent = outputDir || 'Output folder clips';

    try {
      const res = await fetch(`/api/clips?dir_path=${encodeURIComponent(outputDir)}`);
      const data = await res.json();

      clipGrid.innerHTML = '';
      clipCountBadge.textContent = data.clips ? data.clips.length : 0;

      if (!data.clips || data.clips.length === 0) {
        clipGrid.innerHTML = '<p class="empty-msg">No cut clips found in output directory yet.</p>';
        return;
      }

      data.clips.forEach((clip) => {
        const card = document.createElement('div');
        card.className = 'clip-card';
        card.innerHTML = `
          <div class="clip-title">🎬 ${clip.filename}</div>
          <div class="clip-meta">Size: ${clip.size_mb} MB</div>
          <button class="btn btn-secondary btn-sm play-clip-btn">▶ Preview Clip</button>
        `;

        card.querySelector('.play-clip-btn').addEventListener('click', () => {
          openModal(clip.filename, `/api/stream-clip?path=${encodeURIComponent(clip.path)}`);
        });

        clipGrid.appendChild(card);
      });
    } catch (e) {
      console.error('Load clips error:', e);
    }
  }

  function openModal(title, videoSrc) {
    modalTitle.textContent = title;
    modalVideoPlayer.src = videoSrc;
    videoModal.classList.remove('hidden');
    modalVideoPlayer.play();
  }

  function closeModal() {
    modalVideoPlayer.pause();
    modalVideoPlayer.src = '';
    videoModal.classList.add('hidden');
  }

  // ══════════════════════════════════════════════════════════════════════
  // Mode View Switcher Logic
  // ══════════════════════════════════════════════════════════════════════
  const navBatchBtn = document.getElementById('navBatchBtn');
  const navDebugBtn = document.getElementById('navDebugBtn');
  const batchView = document.getElementById('batchView');
  const debugView = document.getElementById('debugView');

  if (navBatchBtn && navDebugBtn) {
    navBatchBtn.addEventListener('click', () => switchView('batchView'));
    navDebugBtn.addEventListener('click', () => switchView('debugView'));
  }

  function switchView(viewId) {
    if (viewId === 'batchView') {
      navBatchBtn.classList.add('active');
      navDebugBtn.classList.remove('active');
      batchView.classList.remove('hidden');
      debugView.classList.add('hidden');
    } else {
      navDebugBtn.classList.add('active');
      navBatchBtn.classList.remove('active');
      debugView.classList.remove('hidden');
      batchView.classList.add('hidden');
    }
  }

  // ══════════════════════════════════════════════════════════════════════
  // Debug Model Inspector Logic
  // ══════════════════════════════════════════════════════════════════════
  const debugVideoPathInput = document.getElementById('debugVideoPath');
  const loadDebugVideoBtn = document.getElementById('loadDebugVideoBtn');
  const debugVideoHint = document.getElementById('debugVideoHint');
  const debugPromptInput = document.getElementById('debugPrompt');
  const debugModelSelect = document.getElementById('debugModelSelect');
  const debugQuantSelect = document.getElementById('debugQuantSelect');
  const debugDualModelCheck = document.getElementById('debugDualModelCheck');
  const debugTextModelGroup = document.getElementById('debugTextModelGroup');
  const debugTextModelSelect = document.getElementById('debugTextModelSelect');
  const debugSimilarityThresholdSlider = document.getElementById('debugSimilarityThresholdSlider');
  const debugSimilarityThresholdValue = document.getElementById('debugSimilarityThresholdValue');
  const debugTimecodeLabel = document.getElementById('debugTimecodeLabel');
  const analyzeFrameBtn = document.getElementById('analyzeFrameBtn');
  const debugMatchBadge = document.getElementById('debugMatchBadge');
  const debugReasonOutput = document.getElementById('debugReasonOutput');
  const debugVideoPlayer = document.getElementById('debugVideoPlayer');

  if (debugDualModelCheck && debugTextModelGroup) {
    debugDualModelCheck.addEventListener('change', () => {
      if (debugDualModelCheck.checked) {
        debugTextModelGroup.classList.remove('hidden');
      } else {
        debugTextModelGroup.classList.add('hidden');
      }
    });
  }

  if (debugSimilarityThresholdSlider && debugSimilarityThresholdValue) {
    debugSimilarityThresholdSlider.addEventListener('input', () => {
      debugSimilarityThresholdValue.textContent = `${debugSimilarityThresholdSlider.value}%`;
    });
  }

  if (loadDebugVideoBtn) {
    loadDebugVideoBtn.addEventListener('click', loadDebugVideo);
  }

  if (debugVideoPlayer) {
    debugVideoPlayer.addEventListener('timeupdate', () => {
      const currentSec = debugVideoPlayer.currentTime || 0;
      const formatted = formatSecondsToTimecode(currentSec);
      if (debugTimecodeLabel) {
        debugTimecodeLabel.textContent = `${formatted} (${currentSec.toFixed(2)}s)`;
      }
    });
  }

  if (analyzeFrameBtn) {
    analyzeFrameBtn.addEventListener('click', analyzeFrame);
  }

  function loadDebugVideo() {
    const videoPath = debugVideoPathInput.value.trim();
    if (!videoPath) {
      alert('Please enter a valid workstation video file path!');
      return;
    }

    const streamUrl = `/api/stream-video?path=${encodeURIComponent(videoPath)}`;
    debugVideoPlayer.src = streamUrl;
    if (debugVideoHint) {
      debugVideoHint.textContent = `✓ Loaded: ${videoPath}`;
      debugVideoHint.style.color = '#34d399';
    }
    if (debugMatchBadge) {
      debugMatchBadge.textContent = 'Not Analyzed';
      debugMatchBadge.className = 'match-badge badge-idle';
    }
    if (debugReasonOutput) {
      debugReasonOutput.textContent = "Position video player to target frame/timecode and click 'Analyze Current Frame / Timestamp' to test prompt.";
    }
  }

  async function analyzeFrame() {
    const videoPath = debugVideoPathInput.value.trim();
    const prompt = debugPromptInput.value.trim();
    const timestamp = debugVideoPlayer.currentTime || 0;
    const isDual = debugDualModelCheck ? debugDualModelCheck.checked : false;

    if (!videoPath) {
      alert('Please specify and load a video file first!');
      return;
    }
    if (!prompt) {
      alert('Please enter a target prompt to analyze!');
      return;
    }

    analyzeFrameBtn.disabled = true;
    analyzeFrameBtn.innerHTML = '<span>⏳</span> Analyzing Frame...';
    debugMatchBadge.textContent = 'Analyzing...';
    debugMatchBadge.className = 'match-badge badge-idle';
    debugReasonOutput.textContent = isDual
      ? `Dual Model: Step 1 (Qwen2-VL describing scene...)`
      : `Analyzing timestamp ${timestamp.toFixed(2)}s with ${debugModelSelect.value}...`;

    try {
      const res = await fetch('/api/debug-frame', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          video_path: videoPath,
          timestamp: timestamp,
          prompt: prompt,
          model_name: debugModelSelect.value,
          quantization: debugQuantSelect.value,
          dual_model: isDual,
          text_model_name: debugTextModelSelect ? debugTextModelSelect.value : null,
          similarity_threshold: debugSimilarityThresholdSlider ? parseFloat(debugSimilarityThresholdSlider.value) : 80.0,
          image_only_micro_batch_size: microBatchSlider ? parseInt(microBatchSlider.value) : 64,
          text_llm_batch_size: textLlmBatchSlider ? parseInt(textLlmBatchSlider.value) : 10,
        }),
      });

      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.detail || 'Analysis failed');
      }

      const data = await res.json();

      if (data.found) {
        debugMatchBadge.textContent = 'TRUE (MATCH)';
        debugMatchBadge.className = 'match-badge badge-true';
      } else {
        debugMatchBadge.textContent = 'FALSE (NO MATCH)';
        debugMatchBadge.className = 'match-badge badge-false';
      }

      const timingVisionLabel = document.getElementById('timingVisionLabel');
      const timingTextLabel = document.getElementById('timingTextLabel');
      const timingTotalLabel = document.getElementById('timingTotalLabel');

      if (timingVisionLabel) timingVisionLabel.textContent = `📷 Vision Model: ${data.vision_time_sec != null ? data.vision_time_sec.toFixed(2) : '--'}s`;
      if (timingTextLabel) timingTextLabel.textContent = `🧠 Text LLM: ${data.text_time_sec != null ? data.text_time_sec.toFixed(2) : '--'}s`;
      if (timingTotalLabel) timingTotalLabel.textContent = `⚡ Total: ${data.total_time_sec != null ? data.total_time_sec.toFixed(2) : '--'}s`;

      if (data.dual_model && data.description) {
        debugReasonOutput.textContent = `--- 🎥 QWEN2-VL SCENE DESCRIPTION ---\n${data.description}\n\n--- 🧠 TEXT LLM CLASSIFICATION REASONING ---\n${data.reason}`;
      } else {
        debugReasonOutput.textContent = data.reason || data.raw_response || 'No response details returned.';
      }

    } catch (e) {
      console.error('Frame analysis error:', e);
      debugMatchBadge.textContent = 'ERROR';
      debugMatchBadge.className = 'match-badge badge-false';
      debugReasonOutput.textContent = `[Error] ${e.message}`;
    } finally {
      analyzeFrameBtn.disabled = false;
      analyzeFrameBtn.innerHTML = '<span>⚡</span> Analyze Current Frame / Timestamp';
    }
  }

  function formatSecondsToTimecode(sec) {
    const hours = Math.floor(sec / 3600);
    const minutes = Math.floor((sec % 3600) / 60);
    const seconds = Math.floor(sec % 60);
    const ms = Math.floor((sec % 1) * 100);

    const pad = (n) => String(n).padStart(2, '0');
    return `${pad(hours)}:${pad(minutes)}:${pad(seconds)}.${pad(ms)}`;
  }

  // ── Dynamic Model Discovery ──

  /**
   * Populate a <select> element with model options.
   * @param {HTMLSelectElement} selectEl - The dropdown to populate
   * @param {Array} models - Array of {id, label} objects
   */
  function populateModelDropdown(selectEl, models) {
    if (!selectEl) return;
    selectEl.innerHTML = '';
    if (models.length === 0) {
      const opt = document.createElement('option');
      opt.value = '';
      opt.textContent = 'No models installed';
      selectEl.appendChild(opt);
      return;
    }
    models.forEach((m, i) => {
      const opt = document.createElement('option');
      opt.value = m.id;
      opt.textContent = m.label;
      if (i === 0) opt.selected = true;
      selectEl.appendChild(opt);
    });
  }

  /**
   * Fetch installed models from the API and populate all 4 model dropdowns.
   */
  async function loadInstalledModels() {
    try {
      const resp = await fetch('/api/models');
      if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
      const data = await resp.json();

      // Batch view dropdowns
      populateModelDropdown(modelSelect, data.vision_models || []);
      populateModelDropdown(textModelSelect, data.text_models || []);

      // Debug view dropdowns
      const debugModelSelect = document.getElementById('debugModelSelect');
      const debugTextModelSelect = document.getElementById('debugTextModelSelect');
      populateModelDropdown(debugModelSelect, data.vision_models || []);
      populateModelDropdown(debugTextModelSelect, data.text_models || []);
    } catch (e) {
      console.error('Failed to load installed models:', e);
      // Leave the "Loading models..." placeholder in place on failure
    }
  }

  // Expose modal functions globally for onclick handlers
  window.openAddModelModal = function() {
    document.getElementById('addModelModal').classList.remove('hidden');
    document.getElementById('downloadModelId').value = '';
    document.getElementById('downloadStatus').style.display = 'none';
    document.getElementById('downloadModelBtn').disabled = false;
  };

  window.closeAddModelModal = function() {
    document.getElementById('addModelModal').classList.add('hidden');
  };

  window.downloadModel = async function() {
    const modelId = document.getElementById('downloadModelId').value.trim();
    const modelType = document.getElementById('downloadModelType').value;
    const statusDiv = document.getElementById('downloadStatus');
    const statusBadge = document.getElementById('downloadStatusBadge');
    const statusMsg = document.getElementById('downloadStatusMsg');
    const downloadBtn = document.getElementById('downloadModelBtn');

    if (!modelId || !modelId.includes('/')) {
      statusDiv.style.display = 'block';
      statusBadge.textContent = 'Error';
      statusBadge.className = 'badge badge-false';
      statusMsg.textContent = 'Please enter a valid model ID (e.g. org/model-name)';
      return;
    }

    // Show downloading state
    statusDiv.style.display = 'block';
    statusBadge.textContent = 'Downloading...';
    statusBadge.className = 'badge badge-processing';
    statusMsg.textContent = `Downloading ${modelId}... This may take several minutes for large models.`;
    downloadBtn.disabled = true;

    try {
      const resp = await fetch('/api/models/download', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ model_id: modelId, model_type: modelType }),
      });

      if (!resp.ok) {
        const errData = await resp.json().catch(() => ({}));
        throw new Error(errData.detail || `HTTP ${resp.status}`);
      }

      const data = await resp.json();

      // Success — refresh all dropdowns
      if (data.models) {
        populateModelDropdown(modelSelect, data.models.vision_models || []);
        populateModelDropdown(textModelSelect, data.models.text_models || []);
        const debugModelSelect = document.getElementById('debugModelSelect');
        const debugTextModelSelect = document.getElementById('debugTextModelSelect');
        populateModelDropdown(debugModelSelect, data.models.vision_models || []);
        populateModelDropdown(debugTextModelSelect, data.models.text_models || []);
      }

      statusBadge.textContent = 'Success';
      statusBadge.className = 'badge badge-true';
      statusMsg.textContent = data.message || `Model ${modelId} downloaded successfully!`;
      downloadBtn.disabled = false;

      // Auto-close modal after 2 seconds
      setTimeout(() => {
        window.closeAddModelModal();
      }, 2000);

    } catch (e) {
      console.error('Model download error:', e);
      statusBadge.textContent = 'Failed';
      statusBadge.className = 'badge badge-false';
      statusMsg.textContent = `Download failed: ${e.message}`;
      downloadBtn.disabled = false;
    }
  };
});
