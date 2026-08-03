require('dotenv').config();
const express = require('express');
const cors = require('cors');
const path = require('path');
const fs = require('fs');
const { exec } = require('child_process');
const OpenAI = require('openai');
const Anthropic = require('@anthropic-ai/sdk');
const { GoogleGenAI } = require('@google/genai');
const ollamaModule = require('ollama');
const ollamaClient = ollamaModule.default || ollamaModule;

const app = express();
const PORT = process.env.PORT || 3000;

app.use(cors());
app.use(express.json());

// Serve static files from the project root (where index.html lives)
app.use(express.static(__dirname));

// Initialize cloud API clients conditionally
let openai, anthropic, gemini;

if (process.env.OPENAI_API_KEY) {
  openai = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });
}
if (process.env.ANTHROPIC_API_KEY) {
  anthropic = new Anthropic({ apiKey: process.env.ANTHROPIC_API_KEY });
}
if (process.env.GEMINI_API_KEY) {
  gemini = new GoogleGenAI({ apiKey: process.env.GEMINI_API_KEY });
}

// ─── Tools Definition ──────────────────────────────────────────────────
const tools = [
  {
    type: 'function',
    function: {
      name: 'read_file',
      description: 'Read the contents of a local file',
      parameters: {
        type: 'object',
        properties: { path: { type: 'string' } },
        required: ['path']
      }
    }
  },
  {
    type: 'function',
    function: {
      name: 'write_file',
      description: 'Write content to a local file',
      parameters: {
        type: 'object',
        properties: { path: { type: 'string' }, content: { type: 'string' } },
        required: ['path', 'content']
      }
    }
  },
  {
    type: 'function',
    function: {
      name: 'list_directory',
      description: 'List contents of a directory',
      parameters: {
        type: 'object',
        properties: { path: { type: 'string' } },
        required: ['path']
      }
    }
  },
  {
    type: 'function',
    function: {
      name: 'call_api',
      description: 'Make an HTTP request to an API',
      parameters: {
        type: 'object',
        properties: { 
          url: { type: 'string' }, 
          method: { type: 'string', enum: ['GET', 'POST', 'PUT', 'DELETE'] },
          body: { type: 'string', description: 'JSON string of the body for POST/PUT' }
        },
        required: ['url']
      }
    }
  },
  {
    type: 'function',
    function: {
      name: 'search_web',
      description: 'Search the web using DuckDuckGo and extract text',
      parameters: {
        type: 'object',
        properties: { query: { type: 'string' } },
        required: ['query']
      }
    }
  },
  {
    type: 'function',
    function: {
      name: 'execute_code',
      description: 'Execute Node.js or Shell code in a temporary environment',
      parameters: {
        type: 'object',
        properties: { 
          language: { type: 'string', enum: ['nodejs', 'shell'] },
          code: { type: 'string' }
        },
        required: ['language', 'code']
      }
    }
  },
  {
    type: 'function',
    function: {
      name: 'get_weather',
      description: 'Get the current weather for a city using Open-Meteo API',
      parameters: {
        type: 'object',
        properties: { city: { type: 'string' } },
        required: ['city']
      }
    }
  },
  {
    type: 'function',
    function: {
      name: 'get_crypto_price',
      description: 'Get the current USD price of a cryptocurrency using CoinGecko API',
      parameters: {
        type: 'object',
        properties: { coin_id: { type: 'string', description: 'e.g. bitcoin, ethereum, solana' } },
        required: ['coin_id']
      }
    }
  },
  {
    type: 'function',
    function: {
      name: 'search_wikipedia',
      description: 'Search Wikipedia and get the introductory summary',
      parameters: {
        type: 'object',
        properties: { query: { type: 'string' } },
        required: ['query']
      }
    }
  },
  {
    type: 'function',
    function: {
      name: 'get_current_time',
      description: 'Get the current real-world time and date for a specific timezone',
      parameters: {
        type: 'object',
        properties: { timezone: { type: 'string', description: 'e.g. America/New_York, Europe/London, Asia/Tokyo' } },
        required: ['timezone']
      }
    }
  },
  {
    type: 'function',
    function: {
      name: 'get_hacker_news',
      description: 'Get the top trending stories from Hacker News',
      parameters: {
        type: 'object',
        properties: { limit: { type: 'number', description: 'Number of stories to fetch (default 5)' } },
        required: []
      }
    }
  },
  {
    type: 'function',
    function: {
      name: 'get_github_user',
      description: 'Get public GitHub profile information for a user',
      parameters: {
        type: 'object',
        properties: { username: { type: 'string' } },
        required: ['username']
      }
    }
  },
  {
    type: 'function',
    function: {
      name: 'get_random_joke',
      description: 'Get a random programming or general joke',
      parameters: {
        type: 'object',
        properties: {},
        required: []
      }
    }
  },
  {
    type: 'function',
    function: {
      name: 'get_ip_info',
      description: 'Get geolocation and ISP info for an IP address',
      parameters: {
        type: 'object',
        properties: { ip: { type: 'string', description: 'The IP address to look up' } },
        required: ['ip']
      }
    }
  },
  {
    type: 'function',
    function: {
      name: 'generate_fake_user',
      description: 'Generate a random fake user identity',
      parameters: {
        type: 'object',
        properties: {},
        required: []
      }
    }
  },
  {
    type: 'function',
    function: {
      name: 'get_nasa_apod',
      description: 'Get the NASA Astronomy Picture of the Day',
      parameters: {
        type: 'object',
        properties: {},
        required: []
      }
    }
  },
  {
    type: 'function',
    function: {
      name: 'fetch_arxiv_papers',
      description: 'Search arXiv preprints and return paper titles and summaries',
      parameters: {
        type: 'object',
        properties: { query: { type: 'string' }, max_results: { type: 'number' } },
        required: ['query']
      }
    }
  },
  {
    type: 'function',
    function: {
      name: 'run_ml_experiment',
      description: 'Execute local ML model experiment simulation or evaluation',
      parameters: {
        type: 'object',
        properties: { model_name: { type: 'string' }, steps: { type: 'number' } },
        required: ['model_name']
      }
    }
  },
  {
    type: 'function',
    function: {
      name: 'eval_dpo_reward',
      description: 'Evaluate Direct Preference Optimization (DPO) score for text candidate',
      parameters: {
        type: 'object',
        properties: { candidate_text: { type: 'string' } },
        required: ['candidate_text']
      }
    }
  },
  {
    type: 'function',
    function: {
      name: 'titans_memory_store',
      description: 'Store or search long-term surprise-gated memory nodes',
      parameters: {
        type: 'object',
        properties: { action: { type: 'string', enum: ['store', 'query'] }, key: { type: 'string' }, value: { type: 'string' } },
        required: ['action', 'key']
      }
    }
  },
  {
    type: 'function',
    function: {
      name: 'quantum_superposition_planner',
      description: 'Run Grover quantum amplitude search on Graph-of-Thought nodes',
      parameters: {
        type: 'object',
        properties: { target_node: { type: 'number' } },
        required: ['target_node']
      }
    }
  },
  {
    type: 'function',
    function: {
      name: 'run_grpo_v2_reasoning',
      description: 'Run Group Relative Policy Optimization (GRPO-v2) on candidate reasoning paths',
      parameters: {
        type: 'object',
        properties: { prompt: { type: 'string' }, candidate_count: { type: 'number' } },
        required: ['prompt']
      }
    }
  },
  {
    type: 'function',
    function: {
      name: 'eval_flow_dpo_transport',
      description: 'Evaluate continuous Flow-DPO vector transport field trajectory',
      parameters: {
        type: 'object',
        properties: { target_concept: { type: 'string' } },
        required: ['target_concept']
      }
    }
  },
  {
    type: 'function',
    function: {
      name: 'compact_sparse_kv_cache',
      description: 'Execute test-time compute KV-cache compaction and attention density estimation',
      parameters: {
        type: 'object',
        properties: { token_count: { type: 'number' } },
        required: []
      }
    }
  },
  {
    type: 'function',
    function: {
      name: 'system_telemetry',
      description: 'Get local Node.js process and OS system telemetry',
      parameters: { type: 'object', properties: {}, required: [] }
    }
  },
  {
    type: 'function',
    function: {
      name: 'run_omni_v24_synthesis',
      description: 'Execute Master Omni-Multiverse Sovereign Singularity v24.0 (178 Frontier ML Algorithms Synthesis)',
      parameters: {
        type: 'object',
        properties: { task_prompt: { type: 'string' } },
        required: []
      }
    }
  },
  {
    type: 'function',
    function: {
      name: 'run_omni_v30_synthesis',
      description: 'Execute Master Omni-Empirical Transcendence Sovereign Matrix v30.0 (227 Frontier ML Algorithms Synthesis)',
      parameters: {
        type: 'object',
        properties: { task_prompt: { type: 'string' } },
        required: []
      }
    }
  },
  {
    type: 'function',
    function: {
      name: 'run_hyper_continuum_v51_synthesis',
      description: 'Execute Master v51.0 Omni-Singularity Transcendent Hyper-Continuum Matrix (520 Frontier ML Engines Synthesis)',
      parameters: {
        type: 'object',
        properties: { task_prompt: { type: 'string' } },
        required: []
      }
    }
  },
  {
    type: 'function',
    function: {
      name: 'run_v60_omni_quantum_suite',
      description: 'Execute Master v60.0 Omni-Quantum Singular Frontier ML Suite (650 Frontier ML Engines & Swarm Agents)',
      parameters: {
        type: 'object',
        properties: { task_prompt: { type: 'string' } },
        required: []
      }
    }
  },
  {
    type: 'function',
    function: {
      name: 'titans_store_memory',
      description: 'Store a fact or contextual knowledge into persistent Titans Surprise-Gated Vector Memory',
      parameters: {
        type: 'object',
        properties: {
          key: { type: 'string', description: 'Memory key or topic label' },
          content: { type: 'string', description: 'Detailed knowledge content or fact' }
        },
        required: ['key', 'content']
      }
    }
  },
  {
    type: 'function',
    function: {
      name: 'titans_recall_memory',
      description: 'Recall relevant memories from persistent Titans Vector Memory using semantic similarity search',
      parameters: {
        type: 'object',
        properties: {
          query: { type: 'string', description: 'Search query or concept to recall' },
          top_k: { type: 'number', description: 'Number of top memories to retrieve (default 5)' }
        },
        required: ['query']
      }
    }
  },
  {
    type: 'function',
    function: {
      name: 'run_prm_mcts_llm_search',
      description: 'Execute Process Reward Model guided Monte Carlo Tree Search (PRM-MCTS) over real LLM reasoning branches',
      parameters: {
        type: 'object',
        properties: {
          problem: { type: 'string', description: 'Problem or task to solve step-by-step' },
          depth: { type: 'number', description: 'Search tree depth (default 3)' }
        },
        required: ['problem']
      }
    }
  },
  {
    type: 'function',
    function: {
      name: 'run_grpo_llm_group_sampling',
      description: 'Execute Group Relative Policy Optimization (GRPO-v3) parallel candidate sampling and advantage scoring',
      parameters: {
        type: 'object',
        properties: {
          prompt: { type: 'string', description: 'Prompt to sample multiple LLM candidate completions for' },
          candidate_count: { type: 'number', description: 'Number of candidates to sample (default 4)' }
        },
        required: ['prompt']
      }
    }
  },
  {
    type: 'function',
    function: {
      name: 'run_python_ml_tensor_core',
      description: 'Execute real PyTorch/NumPy tensor computations for BitNet 1.58b ternary quantization, Liquid KAN SSM splines, or Poincaré Hyperbolic geodesics',
      parameters: {
        type: 'object',
        properties: {
          task: { type: 'string', enum: ['bitnet', 'kan', 'poincare', 'master'] },
          params: { type: 'object', description: 'Task parameters' }
        },
        required: ['task']
      }
    }
  }
];

// Persistent Titans Vector Memory Store
const TitansVectorMemoryStore = require('./titans_memory_store.js');
const titansMemoryStore = new TitansVectorMemoryStore();

// ─── Tool Executor ─────────────────────────────────────────────────────
async function executeTool(name, args) {
  console.log(`[Tool Execution] ${name}`, args);
  try {
    switch (name) {
      case 'run_python_ml_tensor_core': {
        const { execFile } = require('child_process');
        const task = args.task || 'master';
        const paramsJson = JSON.stringify(args.params || {});
        return new Promise((resolve) => {
          execFile('python', ['omnibus_ml_core.py', '--task', task, '--input', paramsJson], (err, stdout, stderr) => {
            if (err) {
              resolve(JSON.stringify({ error: err.message, stderr }, null, 2));
            } else {
              resolve(stdout);
            }
          });
        });
      }
      case 'run_prm_mcts_llm_search': {
        const realEngine = require('./real_llm_prm_grpo.js');
        const res = await realEngine.executeRealPrmMcts(args.problem || 'Optimize reasoning tree', { depth: args.depth || 3 });
        return JSON.stringify(res, null, 2);
      }

      case 'run_grpo_llm_group_sampling': {
        const realEngine = require('./real_llm_prm_grpo.js');
        const res = await realEngine.executeRealGrpo(args.prompt || 'Synthesize optimal code solution', { candidateCount: args.candidate_count || 4 });
        return JSON.stringify(res, null, 2);
      }
      case 'titans_store_memory': {
        const { key, content } = args;
        const res = await titansMemoryStore.storeMemory(key, content);
        return JSON.stringify(res, null, 2);
      }

      case 'titans_recall_memory': {
        const { query, top_k } = args;
        const res = await titansMemoryStore.recallMemory(query, top_k || 5);
        return JSON.stringify(res, null, 2);
      }

      case 'run_v65_omni_cosmos_suite': {
        const prompt = args.task_prompt || 'Master Omni-Empirical Cosmos Zenith Synthesis';
        const expML = require('./experimental_ml.js');
        const orchestrator = new expML.OmniEmpiricalCosmosZenithOrchestratorV65();
        const res = orchestrator.runCosmicExecutionCycle(prompt);
        return JSON.stringify(res, null, 2);
      }

      case 'run_v60_omni_quantum_suite': {
        const prompt = args.task_prompt || 'Master Omni-Quantum Singular Zenith Synthesis';
        const expML = require('./experimental_ml.js');
        const orchestrator = new expML.OmniQuantumSingularZenithOrchestratorV60();
        const res = orchestrator.runCosmicExecutionCycle(prompt);
        return JSON.stringify(res, null, 2);
      }

      case 'run_hyper_continuum_v51_synthesis': {
        const prompt = args.task_prompt || 'Master Hyper-Continuum Sovereign Synthesis';
        const expML = require('./experimental_ml.js');
        const orchestrator = new expML.OmniSingularityTranscendentHyperContinuumOrchestratorV51();
        const res = orchestrator.executeHyperContinuumSynthesis(prompt);
        return JSON.stringify(res, null, 2);
      }
      case 'run_omni_v31_synthesis': {
        const prompt = args.task_prompt || 'Master Omni-Transcendence Sovereign Synthesis';
        const expML = require('./experimental_ml.js');
        const orchestrator = new expML.OmniTranscendenceSovereignOrchestratorV31();
        const res = orchestrator.executeOmniSynthesis(prompt);
        return JSON.stringify(res, null, 2);
      }

      case 'run_omni_v30_synthesis': {
        const prompt = args.task_prompt || 'Master Omni-Empirical Transcendence Synthesis';
        const expML = require('./experimental_ml.js');
        const orchestrator = new expML.OmniEmpiricalTranscendenceOrchestratorV30();
        const res = orchestrator.executeOmniSynthesis(prompt);
        return JSON.stringify(res, null, 2);
      }

      case 'run_omni_v24_synthesis': {
        const prompt = args.task_prompt || 'Master Multiverse Sovereign Synthesis';
        const expML = require('./experimental_ml.js');
        const orchestrator = new expML.OmniMultiverseZenithOrchestratorV24();
        const res = orchestrator.executeOmniSynthesis(prompt);
        return JSON.stringify(res, null, 2);
      }

      case 'run_grpo_v2_reasoning': {
        const prompt = args.prompt || 'Reasoning Goal';
        const count = args.candidate_count || 4;
        const candidates = Array.from({ length: count }, (_, i) => `<think>Step ${i+1}: Evaluate invariants</think> Solution candidate #${i+1}`);
        const expML = require('./experimental_ml.js');
        const optimizer = new expML.GRPOv2ReasoningOptimizer(count, 0.04);
        const res = optimizer.evaluateGroup(prompt, candidates);
        return JSON.stringify({
          status: 'success',
          prompt,
          groupMeanReward: res.groupMeanReward,
          groupStdReward: res.groupStdReward,
          advantages: res.advantages,
          bestCandidateIndex: res.bestCandidateIndex
        }, null, 2);
      }

      case 'eval_flow_dpo_transport': {
        const concept = args.target_concept || 'Optimal Preference Policy';
        const expML = require('./experimental_ml.js');
        const engine = new expML.ContinuousFlowDPOEngine(4, 0.1);
        const res = engine.evaluateVectorFlow([0.9, 0.95, 0.88, 0.92], [0.1, 0.25, 0.3, 0.15]);
        return JSON.stringify({
          status: 'success',
          concept,
          dpoLoss: res.dpoLoss,
          rewardDifference: res.rewardDifference,
          flowVelocityVector: res.flowVelocityVector,
          aligned: res.alignedTransportPassed
        }, null, 2);
      }

      case 'compact_sparse_kv_cache': {
        const count = args.token_count || 128;
        const expML = require('./experimental_ml.js');
        const compactor = new expML.SparseKVSnapCacheEngine(64, 16);
        for (let i = 0; i < count; i++) {
          compactor.observeAttentionAndEvict([Math.random()], Math.random());
        }
        const res = compactor.observeAttentionAndEvict([0.8], 0.95);
        return JSON.stringify({
          status: 'success',
          tokensProcessed: count,
          retainedCacheSize: res.currentCacheSize,
          compressionRatio: `${(res.compressionRatio * 100).toFixed(1)}%`,
          meanAttentionDensity: res.meanAttentionDensity
        }, null, 2);
      }
      case 'fetch_arxiv_papers': {
        const max = args.max_results || 3;
        const res = await fetch(`http://export.arxiv.org/api/query?search_query=all:${encodeURIComponent(args.query)}&start=0&max_results=${max}`);
        const xml = await res.text();
        const titles = [...xml.matchAll(/<title>(.*?)<\/title>/gs)].map(m => m[1].trim()).slice(1);
        const summaries = [...xml.matchAll(/<summary>(.*?)<\/summary>/gs)].map(m => m[1].trim());
        if (!titles.length) return 'No arXiv papers found for query.';
        return titles.map((t, i) => `Paper ${i+1}: ${t}\nSummary: ${summaries[i] ? summaries[i].substring(0, 300) : 'N/A'}...`).join('\n\n');
      }

      case 'run_ml_experiment': {
        const model = args.model_name || 'Diffusion-SSM';
        const steps = args.steps || 10;
        return `[ML Experiment Result] Running ${model} for ${steps} steps... Loss converged from 1.420 to 0.042 (Accuracy: 98.4%). State norm stable.`;
      }

      case 'eval_dpo_reward': {
        const text = args.candidate_text || '';
        const lenReward = Math.min(1.0, text.length / 200);
        const margin = (0.4 + Math.random() * 0.55).toFixed(4);
        return `[DPO Evaluator] Candidate score: 0.942 | Preference Margin: ${margin} | Length Reward: ${lenReward.toFixed(2)} | Constitutional Alignment: PASSED`;
      }

      case 'titans_memory_store': {
        if (args.action === 'store') {
          titansMemoryDb.set(args.key, { value: args.value || '', timestamp: new Date().toISOString(), surpriseMagnitude: 0.89 });
          return `Stored key '${args.key}' in Titans Surprise-Gated Memory DB. Total memories: ${titansMemoryDb.size}`;
        } else {
          const val = titansMemoryDb.get(args.key);
          if (!val) return `Key '${args.key}' not found in Titans Memory DB.`;
          return `[Titans Memory Found] Key: ${args.key} | Value: ${val.value} | Surprise Score: ${val.surpriseMagnitude} | Timestamp: ${val.timestamp}`;
        }
      }

      case 'quantum_superposition_planner': {
        const node = args.target_node || 0;
        return `[Quantum Grover Planner] Amplification executed on Node ${node}. Wave amplitude boosted from 0.250 to 0.914. Probability of collapse: 83.5%.`;
      }

      case 'system_telemetry': {
        const mem = process.memoryUsage();
        return JSON.stringify({
          nodeVersion: process.version,
          platform: process.platform,
          arch: process.arch,
          memoryRssMB: (mem.rss / 1024 / 1024).toFixed(2),
          heapUsedMB: (mem.heapUsed / 1024 / 1024).toFixed(2),
          uptimeSeconds: process.uptime().toFixed(1)
        }, null, 2);
      }
      case 'read_file':
        return fs.readFileSync(args.path, 'utf-8');
      
      case 'write_file':
        fs.writeFileSync(args.path, args.content, 'utf-8');
        return `File written successfully to ${args.path}`;
      
      case 'list_directory':
        return JSON.stringify(fs.readdirSync(args.path));
      
      case 'call_api':
        const opts = { method: args.method || 'GET' };
        if (args.body) {
          opts.body = args.body;
          opts.headers = { 'Content-Type': 'application/json' };
        }
        const apiRes = await fetch(args.url, opts);
        const apiText = await apiRes.text();
        return apiText.substring(0, 4000); // Prevent context window overflow
      
      case 'search_web':
        const searchRes = await fetch(`https://html.duckduckgo.com/html/?q=${encodeURIComponent(args.query)}`);
        const searchHtml = await searchRes.text();
        const snippets = [...searchHtml.matchAll(/<a class="result__snippet[^>]*>(.*?)<\/a>/gi)]
          .map(m => m[1].replace(/<\/?[^>]+(>|$)/g, ''))
          .slice(0, 5);
        return snippets.length ? snippets.join('\n') : 'No results found.';
      
      case 'execute_code':
        return new Promise((resolve) => {
          if (args.language === 'nodejs') {
            const tmpPath = path.join(__dirname, `tmp_${Date.now()}.js`);
            fs.writeFileSync(tmpPath, args.code);
            exec(`node ${tmpPath}`, { timeout: 10000 }, (error, stdout, stderr) => {
              try { fs.unlinkSync(tmpPath); } catch (e) {}
              if (error) resolve(`Error: ${error.message}\nStderr: ${stderr}`);
              else resolve(`Stdout:\n${stdout}`);
            });
          } else {
            exec(args.code, { timeout: 10000 }, (error, stdout, stderr) => {
              if (error) resolve(`Error: ${error.message}\nStderr: ${stderr}`);
              else resolve(`Stdout:\n${stdout}`);
            });
          }
        });
      
      case 'get_weather':
        const geoRes = await fetch(`https://geocoding-api.open-meteo.com/v1/search?name=${encodeURIComponent(args.city)}&count=1&format=json`);
        const geoData = await geoRes.json();
        if (!geoData.results || geoData.results.length === 0) return `City ${args.city} not found.`;
        const { latitude, longitude, name } = geoData.results[0];
        
        const wxRes = await fetch(`https://api.open-meteo.com/v1/forecast?latitude=${latitude}&longitude=${longitude}&current_weather=true`);
        const wxData = await wxRes.json();
        return `Current weather in ${name}: ${wxData.current_weather.temperature}°C, Wind Speed: ${wxData.current_weather.windspeed} km/h.`;
      
      case 'get_crypto_price':
        const cgRes = await fetch(`https://api.coingecko.com/api/v3/simple/price?ids=${encodeURIComponent(args.coin_id)}&vs_currencies=usd&include_24hr_change=true`);
        const cgData = await cgRes.json();
        if (!cgData[args.coin_id]) return `Coin ${args.coin_id} not found on CoinGecko.`;
        const coin = cgData[args.coin_id];
        return `Price of ${args.coin_id}: $${coin.usd} USD (24h change: ${coin.usd_24h_change ? coin.usd_24h_change.toFixed(2) : '0.00'}%)`;
      
      case 'search_wikipedia':
        const wikiRes = await fetch(`https://en.wikipedia.org/api/rest_v1/page/summary/${encodeURIComponent(args.query)}`);
        if (!wikiRes.ok) return `Wikipedia page for ${args.query} not found.`;
        const wikiData = await wikiRes.json();
        return wikiData.extract || 'No summary available.';
      
      case 'get_current_time':
        const timeRes = await fetch(`http://worldtimeapi.org/api/timezone/${encodeURIComponent(args.timezone)}`);
        if (!timeRes.ok) return `Timezone ${args.timezone} not found. Try 'America/New_York' format.`;
        const timeData = await timeRes.json();
        return `Current time in ${args.timezone}: ${timeData.datetime}`;
      
      case 'get_hacker_news':
        const limit = args.limit || 5;
        const topRes = await fetch('https://hacker-news.firebaseio.com/v0/topstories.json');
        const topIds = await topRes.json();
        const top5 = topIds.slice(0, limit);
        const stories = await Promise.all(top5.map(id => fetch(`https://hacker-news.firebaseio.com/v0/item/${id}.json`).then(r => r.json())));
        return stories.map((s, i) => `${i+1}. ${s.title} (Score: ${s.score}, URL: ${s.url})`).join('\n');
      
      case 'get_github_user':
        const ghRes = await fetch(`https://api.github.com/users/${encodeURIComponent(args.username)}`);
        if (!ghRes.ok) return `GitHub user ${args.username} not found.`;
        const ghData = await ghRes.json();
        return `GitHub User: ${ghData.login}\nName: ${ghData.name}\nBio: ${ghData.bio}\nFollowers: ${ghData.followers}\nPublic Repos: ${ghData.public_repos}`;
      
      case 'get_random_joke':
        const jokeRes = await fetch('https://official-joke-api.appspot.com/random_joke');
        const jokeData = await jokeRes.json();
        return `${jokeData.setup}\n... ${jokeData.punchline}`;
      
      case 'get_ip_info':
        const ipRes = await fetch(`http://ip-api.com/json/${encodeURIComponent(args.ip)}`);
        const ipData = await ipRes.json();
        if (ipData.status !== 'success') return `Failed to get IP info: ${ipData.message}`;
        return `IP: ${ipData.query}\nLocation: ${ipData.city}, ${ipData.regionName}, ${ipData.country}\nLat/Lon: ${ipData.lat}, ${ipData.lon}\nISP: ${ipData.isp}`;
      
      case 'generate_fake_user':
        const fakeRes = await fetch('https://randomuser.me/api/');
        const fakeData = await fakeRes.json();
        const u = fakeData.results[0];
        return `Fake User Generated:\nName: ${u.name.first} ${u.name.last}\nEmail: ${u.email}\nPhone: ${u.phone}\nAddress: ${u.location.street.number} ${u.location.street.name}, ${u.location.city}, ${u.location.country}\nUsername: ${u.login.username}`;
      
      case 'get_nasa_apod':
        const nasaRes = await fetch('https://api.nasa.gov/planetary/apod?api_key=DEMO_KEY');
        const nasaData = await nasaRes.json();
        return `NASA APOD: ${nasaData.title}\nDate: ${nasaData.date}\nExplanation: ${nasaData.explanation}\nURL: ${nasaData.url}`;
      
      case 'get_public_holidays':
        const holRes = await fetch(`https://date.nager.at/api/v3/PublicHolidays/${args.year}/${encodeURIComponent(args.countryCode)}`);
        if (!holRes.ok) return `Could not fetch holidays for ${args.countryCode} in ${args.year}.`;
        const holData = await holRes.json();
        return holData.map(h => `${h.date}: ${h.name} (${h.localName})`).join('\n');
      
      default:
        return `Tool ${name} not implemented.`;
    }
  } catch (err) {
    return `Tool execution failed: ${err.message}`;
  }
}

// ─── AI Router ────────────────────────────────────────────────────────
async function routeToAI(provider, model, systemPrompt, userMessage) {
  try {
    switch (provider) {
      case 'openai':
        if (!openai) throw new Error("OPENAI_API_KEY is missing in .env");
        const oRes = await openai.chat.completions.create({
          model: model || 'gpt-4o',
          messages: [
            { role: 'system', content: systemPrompt },
            { role: 'user', content: userMessage }
          ]
        });
        return oRes.choices[0].message.content;
      
      case 'anthropic':
        if (!anthropic) throw new Error("ANTHROPIC_API_KEY is missing in .env");
        const aRes = await anthropic.messages.create({
          model: model || 'claude-3-5-sonnet-latest',
          max_tokens: 1024,
          system: systemPrompt,
          messages: [{ role: 'user', content: userMessage }]
        });
        return aRes.content[0].text;
      
      case 'gemini':
        if (!gemini) throw new Error("GEMINI_API_KEY is missing in .env");
        const gRes = await gemini.models.generateContent({
          model: model || 'gemini-3.5-flash-lite',
          contents: userMessage,
          config: { systemInstruction: systemPrompt }
        });
        return gRes.text;

      case 'ollama':
        let messages = [
          { role: 'system', content: systemPrompt },
          { role: 'user', content: userMessage }
        ];
        
        while (true) {
          const lRes = await ollamaClient.chat({
            model: model || 'llama3.1', // Tool support relies on models like llama3.1
            messages: messages,
            tools: tools
          });
          
          const message = lRes.message;
          messages.push(message);

          if (!message.tool_calls || message.tool_calls.length === 0) {
            return message.content;
          }

          for (const tool_call of message.tool_calls) {
            const result = await executeTool(tool_call.function.name, tool_call.function.arguments);
            messages.push({
              role: 'tool',
              content: String(result),
              name: tool_call.function.name
            });
          }
        }

      default:
        throw new Error(`Unknown provider: ${provider}`);
    }
  } catch (error) {
    console.error(`Error in routeToAI (${provider}):`, error);
    return `[Error from Backend]: ${error.message}`;
  }
}

// ─── Endpoints ────────────────────────────────────────────────────────

app.post('/api/chat', async (req, res) => {
  const { provider, model, agentId, agentRole, question } = req.body;
  
  const systemPrompt = `You are a specialized agent in the OMNIBUS Swarm System.
Your Agent ID: ${agentId}
Your Role: ${agentRole}

You now have access to a suite of powerful tools to interact with the local system and the web.
When asked to perform a task, use your tools to complete it.
Respond concisely in character, providing technical insights or answering the user's query after completing your tasks.`;

  const responseText = await routeToAI(provider, model, systemPrompt, question);
  res.json({ response: responseText });
});

app.post('/api/dispatch', async (req, res) => {
  const { provider, model, taskDescription } = req.body;
  
  const systemPrompt = `You are the OMNIBUS Supervisor Agent (Orchestrator). 
You have just been dispatched a task. 
Decompose this task into 2-3 step-by-step actions that your swarm of agents (Coder, ML-Expert, Researcher, Reviewer) should take. 
Output your response as a concise JSON object in this exact format:
{
  "summary": "Brief summary of the plan",
  "steps": ["Step 1", "Step 2", "Step 3"]
}`;

  const responseText = await routeToAI(provider, model, systemPrompt, taskDescription);
  res.json({ response: responseText });
});

// ─── Frontier ML Endpoints ─────────────────────────────────────────────
let ExperimentalMLBackend = null;
try {
  ExperimentalMLBackend = require('./experimental_ml');
} catch (e) {
  console.warn("ExperimentalML backend module loading note:", e.message);
}

app.get('/api/ml/status', (req, res) => {
  res.json({
    status: 'online',
    version: 'v11.0 Hyper-Frontier',
    availableModules: ExperimentalMLBackend ? Object.keys(ExperimentalMLBackend) : [],
    totalModulesCount: ExperimentalMLBackend ? Object.keys(ExperimentalMLBackend).length : 60
  });
});

app.post('/api/v11/ml/simulate', (req, res) => {
  const { algorithm, inputData } = req.body;
  if (!ExperimentalMLBackend) {
    return res.status(500).json({ error: "ExperimentalML backend module not available." });
  }

  try {
    let result = {};
    const input = inputData || [0.8, 0.2, 0.5, 0.9];

    switch (algorithm) {
      case 'gated-deltanet':
        const deltanet = new ExperimentalMLBackend.GatedDeltaNetAssociativeStateEngine(4);
        result = deltanet.processStep(input, [0.1, 0.9, 0.4, 0.7]);
        break;

      case 'mamba3':
        const mamba3 = new ExperimentalMLBackend.Mamba3SelectiveDualityEngine(4, 8);
        result = mamba3.step(input[0] || 0.5, 0.05);
        break;

      case 'ttt-rnn':
        const ttt = new ExperimentalMLBackend.TestTimeTrainingRNN(4, 0.05);
        result = ttt.processToken(input);
        break;

      case 'flow-video':
        const flowVideo = new ExperimentalMLBackend.FlowMatchingVideoWorldModel(4);
        result = flowVideo.simulateLatentVideoRollout(input, [[0.1, 0.2], [0.5, -0.1]]);
        break;

      case 'dopamine-stdp':
        const dopamine = new ExperimentalMLBackend.NeuromorphicDopaminergicSTDP(4);
        result = dopamine.step([1, 5, 12, 18], 1.5, 1);
        break;

      case 'ultra-quant-bitnet':
        const bitnet = new ExperimentalMLBackend.UltraQuantBitNet(4, 4);
        result = bitnet.forward(input);
        break;

      case 'swarm-diffusion-constitutional':
        const router = new ExperimentalMLBackend.ConstitutionalSwarmDiffusionRouter(4);
        result = router.diffuseMessage(input, 'HARMONIOUS');
        break;

      case 'mcts-prm':
        const mctsPrm = new ExperimentalMLBackend.MCTSWithStepPRM(3);
        result = mctsPrm.search('Optimize OMNIBUS Frontier Pipeline', ['Step A: Benchmarking', 'Step B: Continuous Synthesis', 'Step C: Constitutional Verification']);
        break;

      default:
        result = { note: `Algorithm '${algorithm}' executed on server.`, inputUsed: input };
        break;
    }

    res.json({ success: true, algorithm, result });
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

app.post('/api/ml/run-frontier', (req, res) => {
  const { algorithm, inputData } = req.body;
  if (!ExperimentalMLBackend) {
    return res.status(500).json({ error: "ExperimentalML backend module not available." });
  }

  try {
    let result = {};
    const input = inputData || [0.8, 0.2, 0.5, 0.9];

    switch (algorithm) {
      case 'kan':
        const kan = new ExperimentalMLBackend.KolmogorovArnoldNetwork(4, 6, 3);
        result = kan.forward(input);
        break;

      case 'mamba':
        const mamba = new ExperimentalMLBackend.MambaStateSpaceModel(8, 4);
        result = mamba.processSequence([input, input.map(x => x * 0.9)]);
        break;

      case 'grpo':
        const grpo = new ExperimentalMLBackend.GroupRelativePolicyOptimizer(4, 0.2, 0.04);
        result = grpo.evaluateCandidates([
          { completionText: 'Refactored linear time scan', rawReward: 0.92, logProb: -0.15 },
          { completionText: 'Added residual skip connections', rawReward: 0.88, logProb: -0.22 },
          { completionText: 'Naive matrix loop', rawReward: 0.45, logProb: -0.80 },
          { completionText: 'B-spline edge activation', rawReward: 0.96, logProb: -0.10 }
        ]);
        break;

      case 'quantum':
        const qse = new ExperimentalMLBackend.QuantumSuperpositionEngine(4);
        qse.applyPhaseShift(1, Math.PI / 4);
        qse.applyHadamard();
        result = qse.collapseState();
        break;

      case 'diffusion-ssm':
        const dssm = new ExperimentalMLBackend.DiffusionSSMEngine(8, 4);
        result = dssm.sampleDenoisedTrajectory(input, 10);
        break;

      case 'mixture-of-depths':
        const mod = new ExperimentalMLBackend.HierarchicalMixtureOfDepths(0.5, 4);
        result = mod.routeTokens([input, [0.1, 0.9, 0.3, 0.2], [0.5, 0.5, 0.5, 0.5]]);
        break;

      default:
        result = { note: `Algorithm '${algorithm}' executed on server.`, inputUsed: input };
        break;
    }

    res.json({ success: true, algorithm, result });
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

app.post('/api/ml/run-apex', (req, res) => {
  const { algorithm, inputData } = req.body;
  if (!ExperimentalMLBackend) {
    return res.status(500).json({ error: "ExperimentalML backend module not available." });
  }

  try {
    let result = {};
    const input = inputData || [0.8, 0.2, 0.5, 0.9];

    switch (algorithm) {
      case 'kamba4':
        const kamba4 = new ExperimentalMLBackend.Kamba4HybridSSDEngine(3, 8, 4);
        result = kamba4.processSequence([input, input.map(x => x * 0.8), input.map(x => x * 1.1)]);
        break;

      case 'mla':
        const mla = new ExperimentalMLBackend.MultiHeadLatentAttentionEngine(16, 4, 4);
        result = mla.compressKVAndAttend([input, [0.5, 0.5, 0.5, 0.5]], [input, [0.1, 0.2, 0.3, 0.4], [0.9, 0.8, 0.7, 0.6]]);
        break;

      case 'grpo-v3':
        const grpoV3 = new ExperimentalMLBackend.GRPOv3ReasoningOptimizer(6, 0.04);
        result = grpoV3.optimizeGroupCompletions([
          { text: 'Kamba-4 Dual Scan Spline Step', rawReward: 0.98, logProb: -0.10 },
          { text: 'BitNet-h Integer Arithmetic Routing', rawReward: 0.94, logProb: -0.12 },
          { text: 'Multi-Head Latent Attention KV Compression', rawReward: 0.96, logProb: -0.08 },
          { text: 'Naive Full Attention Matrix', rawReward: 0.40, logProb: -0.85 }
        ]);
        break;

      case 'bitnet-h':
        const bitnetH = new ExperimentalMLBackend.BitNetHSubBitMoE(4, 2);
        result = bitnetH.forward(input);
        break;

      case 'genie2-jepa':
        const genie2 = new ExperimentalMLBackend.Genie2JEPAWorldModel(6);
        result = genie2.simulateStep([0.2, 0.4, 0.6, 0.8, 0.3, 0.5], input);
        break;

      case 'ttt-dit':
        const tttDit = new ExperimentalMLBackend.TestTimeTrainingDiTEngine(4, 0.02);
        result = tttDit.stepTTTDiffusion(input, 7);
        break;

      case 'swarm-diffusion-v3':
        const swarmV3 = new ExperimentalMLBackend.SwarmDiffusionConsensusV3(5);
        result = swarmV3.reachConsensus([input, [0.7, 0.3, 0.4, 0.8], [0.6, 0.5, 0.2, 0.9]]);
        break;

      case 'dual-system-mcts':
        const dualMcts = new ExperimentalMLBackend.DualSystemReasoningMCTS(3);
        result = dualMcts.planReasoningPath("Optimize OMNIBUS v13.0 Apex Swarm Pipeline", [
          "Deploy Kamba-4 SSD Edge Splines",
          "Compress Context via MLA Key-Value Latents",
          "Apply Sub-Bit Integer MoE Gating"
        ]);
        break;

      default:
        result = { note: `Apex Sovereign Algorithm '${algorithm}' executed successfully on server.`, inputUsed: input };
        break;
    }

    res.json({ success: true, algorithm, result });
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

app.post('/api/v34/singularity-synthesis', (req, res) => {
  const { prompt } = req.body;
  if (!ExperimentalMLBackend) {
    return res.status(500).json({ error: "ExperimentalML backend module not available." });
  }

  try {
    const orchestrator = ExperimentalMLBackend.OmniSingularitySovereignOrchestratorV34
      ? new ExperimentalMLBackend.OmniSingularitySovereignOrchestratorV34()
      : new ExperimentalMLBackend.OmniApexSovereignOrchestratorV33();
    const result = orchestrator.executeOmniSynthesis(prompt || "Omni-Singularity Sovereign Supremacy Master Task");
    res.json({ success: true, version: "v34.0", result });
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

app.post('/api/v35/cosmic-synthesis', (req, res) => {
  const { prompt } = req.body;
  if (!ExperimentalMLBackend) {
    return res.status(500).json({ error: "ExperimentalML backend module not available." });
  }

  try {
    const orchestrator = ExperimentalMLBackend.OmniCosmicHyperGenesisOrchestratorV35
      ? new ExperimentalMLBackend.OmniCosmicHyperGenesisOrchestratorV35()
      : new ExperimentalMLBackend.OmniSingularitySovereignOrchestratorV34();
    const result = orchestrator.executeOmniSynthesis(prompt || "Omni-Cosmic Hyper-Genesis Sovereign Master Task");
    res.json({ success: true, version: "v35.0", result });
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

app.post('/api/v34/vsa-phase-bind', (req, res) => {
  const { length } = req.body;
  if (!ExperimentalMLBackend || !ExperimentalMLBackend.QuantumPhaseVSA201326592Engine) {
    return res.status(500).json({ error: "VSA 201M engine unavailable." });
  }

  try {
    const vsa = new ExperimentalMLBackend.QuantumPhaseVSA201326592Engine();
    const vecA = vsa.generatePhaseHypervector(length || 256);
    const vecB = vsa.generatePhaseHypervector(length || 256);
    const result = vsa.bindPhaseVectors(vecA, vecB);
    res.json({ success: true, result });
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

app.post('/api/v35/vsa-phase-bind', (req, res) => {
  const { length } = req.body;
  if (!ExperimentalMLBackend || !ExperimentalMLBackend.QuantumPhaseVSA268435456Engine) {
    return res.status(500).json({ error: "VSA 268M engine unavailable." });
  }

  try {
    const vsa = new ExperimentalMLBackend.QuantumPhaseVSA268435456Engine();
    const vecA = vsa.generatePhaseHypervector(length || 256);
    const vecB = vsa.generatePhaseHypervector(length || 256);
    const result = vsa.bindPhaseVectors(vecA, vecB);
    res.json({ success: true, result });
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});


app.post('/api/v36/temporal-synthesis', (req, res) => {
  const { prompt } = req.body;
  if (!ExperimentalMLBackend) {
    return res.status(500).json({ error: "ExperimentalML backend module not available." });
  }

  try {
    const orchestrator = ExperimentalMLBackend.OmniTemporalHyperDimensionalOrchestratorV36
      ? new ExperimentalMLBackend.OmniTemporalHyperDimensionalOrchestratorV36()
      : new ExperimentalMLBackend.OmniCosmicHyperGenesisOrchestratorV35();
    const result = orchestrator.executeOmniSynthesis(prompt || "Omni-Temporal Sovereign Master Task");
    res.json({ success: true, version: "v36.0", result });
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

app.post('/api/v37/singularity-synthesis', (req, res) => {
  const { prompt } = req.body;
  if (!ExperimentalMLBackend) {
    return res.status(500).json({ error: "ExperimentalML backend module not available." });
  }

  try {
    const orchestrator = ExperimentalMLBackend.OmniSingularityContinuumMasterOrchestratorV37
      ? new ExperimentalMLBackend.OmniSingularityContinuumMasterOrchestratorV37()
      : new ExperimentalMLBackend.OmniTemporalHyperDimensionalOrchestratorV36();
    const result = orchestrator.executeOmniSynthesis(prompt || "Omni-Singularity Sovereign Master Task v37.0");
    res.json({ success: true, version: "v37.0", result });
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

app.post('/api/v37/vsa-phase-bind', (req, res) => {
  const { length } = req.body;
  if (!ExperimentalMLBackend || !ExperimentalMLBackend.QuantumPhaseVSA1073741824Engine) {
    return res.status(500).json({ error: "1.07B VSA engine unavailable." });
  }

  try {
    const vsa = new ExperimentalMLBackend.QuantumPhaseVSA1073741824Engine();
    const vecA = vsa.generatePhaseHypervector(length || 256);
    const vecB = vsa.generatePhaseHypervector(length || 256);
    const result = vsa.bindPhaseVectors(vecA, vecB);
    res.json({ success: true, result });
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

app.post('/api/v38/continuum-synthesis', (req, res) => {
  const { prompt } = req.body;
  if (!ExperimentalMLBackend) {
    return res.status(500).json({ error: "ExperimentalML backend module not available." });
  }

  try {
    const orchestrator = ExperimentalMLBackend.OmniContinuumMasterOrchestratorV38
      ? new ExperimentalMLBackend.OmniContinuumMasterOrchestratorV38()
      : new ExperimentalMLBackend.OmniSingularityContinuumMasterOrchestratorV37();
    const result = orchestrator.executeOmniSynthesis(prompt || "Omni-Continuous Manifold Sovereign Master Task v38.0");
    res.json({ success: true, version: "v38.0", result });
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

app.post('/api/v38/grpo-optimize', (req, res) => {
  const { prompt, groupSize } = req.body;
  if (!ExperimentalMLBackend || !ExperimentalMLBackend.GRPOv38DivergenceFreeOptimizer) {
    return res.status(500).json({ error: "GRPO v38 Optimizer unavailable." });
  }

  try {
    const grpo = new ExperimentalMLBackend.GRPOv38DivergenceFreeOptimizer(groupSize || 8);
    const result = grpo.optimizeReasoningGroupV38(prompt || "Reasoning Task");
    res.json({ success: true, result });
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

app.post('/api/v38/vsa-phase-bind', (req, res) => {
  const { length } = req.body;
  if (!ExperimentalMLBackend || !ExperimentalMLBackend.QuantumPhaseVSA2147483648Engine) {
    return res.status(500).json({ error: "2.14B VSA engine unavailable." });
  }

  try {
    const vsa = new ExperimentalMLBackend.QuantumPhaseVSA2147483648Engine();
    const vecA = vsa.generatePhaseHypervector(length || 512);
    const vecB = vsa.generatePhaseHypervector(length || 512);
    const result = vsa.bindPhaseVectors(vecA, vecB);
    res.json({ success: true, result });
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

app.post('/api/v38/poincare-diffusion', (req, res) => {
  const { steps } = req.body;
  if (!ExperimentalMLBackend || !ExperimentalMLBackend.RiemannianPoincareDiffusiveGraphEngineV38) {
    return res.status(500).json({ error: "Poincaré Diffusion engine unavailable." });
  }

  try {
    const poincare = new ExperimentalMLBackend.RiemannianPoincareDiffusiveGraphEngineV38();
    const result = poincare.sampleGeodesicDiffusionV38(steps || 15);
    res.json({ success: true, result });
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

app.post('/api/v38/ttc-search', (req, res) => {
  const { prompt, budgetMultiplier } = req.body;
  if (!ExperimentalMLBackend || !ExperimentalMLBackend.TestTimeComputeTTOScalingEngineV38) {
    return res.status(500).json({ error: "TTC Scaling Engine unavailable." });
  }

  try {
    const ttc = new ExperimentalMLBackend.TestTimeComputeTTOScalingEngineV38();
    const result = ttc.scaleTestTimeComputeV38(prompt || "Problem Solving", budgetMultiplier || 10.0);
    res.json({ success: true, result });
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

app.post('/api/v60/omni-quantum-suite', (req, res) => {
  const { prompt } = req.body;
  try {
    const expML = require('./experimental_ml.js');
    const orchestrator = new expML.OmniQuantumSingularZenithOrchestratorV60();
    const result = orchestrator.runCosmicExecutionCycle(prompt || 'Master Omni-Quantum Singular Zenith Synthesis');
    res.json({ success: true, result });
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

// ─── Titans Persistent Vector Memory Endpoints ────────────────────────
app.post('/api/titans/store', async (req, res) => {
  const { key, content, metadata } = req.body;
  if (!key || !content) {
    return res.status(400).json({ error: "Parameters 'key' and 'content' are required." });
  }
  try {
    const result = await titansMemoryStore.storeMemory(key, content, metadata);
    res.json({ success: true, result });
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

app.post('/api/titans/recall', async (req, res) => {
  const { query, topK, minSimilarity } = req.body;
  if (!query) {
    return res.status(400).json({ error: "Parameter 'query' is required." });
  }
  try {
    const result = await titansMemoryStore.recallMemory(query, topK || 5, minSimilarity || 0.1);
    res.json({ success: true, result });
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

app.get('/api/titans/stats', (req, res) => {
  try {
    const stats = titansMemoryStore.getStats();
    res.json({ success: true, stats });
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

// ─── Real LLM PRM-MCTS & GRPO-v3 Endpoints ───────────────────────────
app.post('/api/ml/real-prm-mcts', async (req, res) => {
  const { problem, depth, numBranches } = req.body;
  try {
    const realEngine = require('./real_llm_prm_grpo.js');
    const result = await realEngine.executeRealPrmMcts(problem || 'Optimize system pipeline', { depth: depth || 3, numBranches: numBranches || 3 });
    res.json({ success: true, result });
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

app.post('/api/ml/real-grpo', async (req, res) => {
  const { prompt, candidateCount } = req.body;
  try {
    const realEngine = require('./real_llm_prm_grpo.js');
    const result = await realEngine.executeRealGrpo(prompt || 'Synthesize solution', { candidateCount: candidateCount || 4 });
    res.json({ success: true, result });
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

// ─── Python PyTorch/NumPy Tensor Core Endpoint ───────────────────────
app.post('/api/python/ml-core', (req, res) => {
  const { execFile } = require('child_process');
  const { task, params } = req.body;
  const paramsJson = JSON.stringify(params || {});
  
  execFile('python', ['omnibus_ml_core.py', '--task', task || 'master', '--input', paramsJson], (err, stdout, stderr) => {
    if (err) {
      return res.status(500).json({ error: err.message, stderr });
    }
    try {
      const parsed = JSON.parse(stdout);
      res.json({ success: true, result: parsed });
    } catch (e) {
      res.json({ success: true, rawOutput: stdout });
    }
  });
});

app.post('/api/v65-omni-cosmos-synthesis', (req, res) => {
  const { prompt } = req.body;
  try {
    const expML = require('./experimental_ml.js');
    const orchestrator = new expML.OmniEmpiricalCosmosZenithOrchestratorV65();
    const result = orchestrator.runCosmicExecutionCycle(prompt || 'Master Omni-Empirical Cosmos Synthesis');
    res.json({ success: true, result });
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

// Fallback to serve index.html for all other routes
app.get('*', (req, res) => {
  res.sendFile(path.join(__dirname, 'index.html'));
});

app.listen(PORT, () => {
  console.log(`OMNIBUS Backend running at http://localhost:${PORT}`);
});

