/**
 * OMNIBUS Single Chat Interface & Backend Controller
 */
document.addEventListener('DOMContentLoaded', () => {
  const chatMessagesStream = document.getElementById('chatMessagesStream');
  const chatInputField = document.getElementById('chatInputField');
  const chatSendBtn = document.getElementById('chatSendBtn');
  const engineModeSelector = document.getElementById('engineModeSelector');
  const clearChatBtn = document.getElementById('clearChatBtn');
  const orbStateLabel = document.getElementById('orbStateLabel');

  // Auto-resize textarea
  if (chatInputField) {
    chatInputField.addEventListener('input', () => {
      chatInputField.style.height = 'auto';
      chatInputField.style.height = Math.min(chatInputField.scrollHeight, 120) + 'px';
    });

    chatInputField.addEventListener('keydown', (e) => {
      if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        handleSendMessage();
      }
    });
  }

  if (chatSendBtn) {
    chatSendBtn.addEventListener('click', handleSendMessage);
  }

  if (clearChatBtn) {
    clearChatBtn.addEventListener('click', () => {
      if (chatMessagesStream) {
        chatMessagesStream.innerHTML = `
          <div class="message-card assistant">
            <div class="avatar assistant"><i class="fas fa-robot"></i></div>
            <div class="message-bubble">
              <div style="font-weight: 800; color: #00f0ff; margin-bottom: 4px;">OMNIBUS Core Assistant</div>
              Chat history cleared. System ready for new queries!
            </div>
          </div>
        `;
      }
    });
  }

  function setOrbState(state, labelText) {
    if (window.orbVisualizer) {
      window.orbVisualizer.setState(state);
    }
    if (orbStateLabel) {
      orbStateLabel.innerText = labelText || (state === 'thinking' ? 'Computing Real LLM Reasoning...' : 'OMNIBUS Neural Interface · Ready');
    }
  }

  function appendUserMessage(text) {
    const card = document.createElement('div');
    card.className = 'message-card user';
    card.innerHTML = `
      <div class="avatar user"><i class="fas fa-user"></i></div>
      <div class="message-bubble">${escapeHtml(text)}</div>
    `;
    chatMessagesStream.appendChild(card);
    scrollToBottom();
  }

  function appendAssistantMessage(htmlContent, title = 'OMNIBUS Neural Core') {
    const card = document.createElement('div');
    card.className = 'message-card assistant';
    card.innerHTML = `
      <div class="avatar assistant"><i class="fas fa-microchip"></i></div>
      <div class="message-bubble">
        <div style="font-weight: 800; color: #00f0ff; margin-bottom: 6px;">${title}</div>
        <div>${htmlContent}</div>
      </div>
    `;
    chatMessagesStream.appendChild(card);
    scrollToBottom();
  }

  function scrollToBottom() {
    if (chatMessagesStream) {
      chatMessagesStream.scrollTop = chatMessagesStream.scrollHeight;
    }
  }

  function escapeHtml(str) {
    return str.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
  }

  async function handleSendMessage() {
    const promptText = chatInputField.value.trim();
    if (!promptText) return;

    chatInputField.value = '';
    chatInputField.style.height = 'auto';
    appendUserMessage(promptText);

    const mode = engineModeSelector ? engineModeSelector.value : 'prm_mcts';
    setOrbState('thinking', 'Computing Real LLM Reasoning...');

    try {
      if (mode === 'prm_mcts') {
        const response = await fetch('/api/ml/real-prm-mcts', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ problem: promptText, depth: 3 })
        });
        const data = await response.json();
        setOrbState('responding', 'Tree-Search Completed');
        renderPrmMctsResult(data.result);
      } 
      else if (mode === 'grpo_v3') {
        const response = await fetch('/api/ml/real-grpo', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ prompt: promptText, candidateCount: 4 })
        });
        const data = await response.json();
        setOrbState('responding', 'GRPO Optimization Complete');
        renderGrpoResult(data.result);
      }
      else if (mode === 'titans') {
        if (promptText.toLowerCase().startsWith('store')) {
          const content = promptText.replace(/^store\s*/i, '');
          const response = await fetch('/api/titans/store', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ key: "User Memory", content: content || promptText })
          });
          const data = await response.json();
          setOrbState('responding', 'Memory Stored');
          renderTitansStoreResult(data.result);
        } else {
          const response = await fetch('/api/titans/recall', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ query: promptText, topK: 3 })
          });
          const data = await response.json();
          setOrbState('responding', 'Memory Recalled');
          renderTitansRecallResult(data.result);
        }
      }
      else if (mode === 'python_core') {
        let task = 'master';
        if (promptText.toLowerCase().includes('bitnet')) task = 'bitnet';
        else if (promptText.toLowerCase().includes('kan')) task = 'kan';
        else if (promptText.toLowerCase().includes('poincare')) task = 'poincare';

        const response = await fetch('/api/python/ml-core', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ task, params: { query: promptText } })
        });
        const data = await response.json();
        setOrbState('responding', 'Python Core Execution Complete');
        renderPythonResult(data.result);
      }

      setTimeout(() => {
        setOrbState('idle', 'OMNIBUS Neural Interface · System Ready');
      }, 1500);

    } catch (err) {
      setOrbState('idle', 'Error Encountered');
      appendAssistantMessage(`<span style="color: #ff0055;">Error: ${err.message}</span>`, 'System Alert');
    }
  }

  function renderPrmMctsResult(res) {
    if (!res) return;
    let pathHtml = res.chosenExecutionPath.map(step => `
      <div style="margin: 6px 0; padding: 8px 12px; background: rgba(0,240,255,0.06); border-left: 3px solid #00f0ff; border-radius: 4px;">
        <span style="color: #00f0ff; font-weight: bold;">[Step Depth ${step.depth}]</span> (Score: ${step.prmScore}): ${escapeHtml(step.stepContent)}
      </div>
    `).join('');

    let html = `
      <div style="font-size: 0.9rem;">
        <div><b>Engine:</b> ${res.engine}</div>
        <div><b>Provider:</b> <span style="color: #00ff88; font-weight: bold;">${res.provider}</span></div>
        <div><b>Pass Rate:</b> <span style="color: #ff00ea; font-weight: bold;">${res.verifiedPassRate}</span></div>
        <div><b>Nodes Evaluated:</b> ${res.totalTreeNodesEvaluated} (Depth: ${res.treeDepth})</div>
        <div style="margin-top: 10px; font-weight: bold; color: #00f0ff;">Verified Optimal Path:</div>
        ${pathHtml}
      </div>
    `;
    appendAssistantMessage(html, '🌲 Real PRM-MCTS Tree Search Result');
  }

  function renderGrpoResult(res) {
    if (!res) return;
    let html = `
      <div style="font-size: 0.9rem;">
        <div><b>Engine:</b> ${res.engine}</div>
        <div><b>Provider:</b> <span style="color: #00ff88; font-weight: bold;">${res.provider}</span></div>
        <div><b>Group Size (K):</b> ${res.groupSize}</div>
        <div><b>Winning Advantage:</b> <span style="color: #ff00ea; font-weight: bold;">${res.winningCandidate?.advantagePercentage}</span></div>
        
        <div class="code-block-wrapper">
          <div class="code-block-header">Winning Reasoning Candidate (Temp = ${res.winningCandidate?.temperature})</div>
          <div class="code-block-content">${escapeHtml(res.winningCandidate?.solutionText || '')}</div>
        </div>
      </div>
    `;
    appendAssistantMessage(html, '🧮 Real GRPO-v3 Relative Advantage Result');
  }

  function renderTitansStoreResult(res) {
    if (!res) return;
    let html = `
      <div style="font-size: 0.9rem;">
        <div><b>Memory Status:</b> <span style="color: #00ff88; font-weight: bold;">STORED IN PERSISTENT VECTOR DB</span></div>
        <div><b>Surprise Gate Metric S:</b> ${res.memory?.surpriseMetric}</div>
        <div><b>Retention Score:</b> ${(res.memory?.retentionScore * 100).toFixed(2)}%</div>
        <div><b>Memory Key:</b> ${escapeHtml(res.memory?.key || '')}</div>
        <div><b>Content:</b> ${escapeHtml(res.memory?.content || '')}</div>
      </div>
    `;
    appendAssistantMessage(html, '🧠 Titans Surprise-Gated Memory Store');
  }

  function renderTitansRecallResult(res) {
    if (!res) return;
    let memoriesHtml = (res.recalledMemories || []).map(m => `
      <div style="margin: 6px 0; padding: 8px 12px; background: rgba(255,0,234,0.06); border-left: 3px solid #ff00ea; border-radius: 4px;">
        <div style="color: #ff00ea; font-weight: bold;">${escapeHtml(m.key)} (Similarity: ${(m.similarity * 100).toFixed(1)}%)</div>
        <div style="color: #e2e8f0; margin-top: 4px;">${escapeHtml(m.content)}</div>
      </div>
    `).join('');

    let html = `
      <div style="font-size: 0.9rem;">
        <div><b>Query:</b> "${escapeHtml(res.query)}"</div>
        <div><b>Matches Found:</b> ${res.resultsCount}</div>
        <div style="margin-top: 10px;">${memoriesHtml || 'No matching vector memories found.'}</div>
      </div>
    `;
    appendAssistantMessage(html, '🧠 Titans Vector Memory Recall');
  }

  function renderPythonResult(res) {
    if (!res) return;
    let jsonFormatted = escapeHtml(JSON.stringify(res, null, 2));
    let html = `
      <div style="font-size: 0.9rem;">
        <div><b>Python PyTorch/NumPy Tensor Core v65.0:</b></div>
        <div class="code-block-wrapper">
          <div class="code-block-header">Python Execution JSON Output</div>
          <div class="code-block-content">${jsonFormatted}</div>
        </div>
      </div>
    `;
    appendAssistantMessage(html, '🐍 Python PyTorch/NumPy Tensor Core Result');
  }
});
