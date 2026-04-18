import React, { useEffect, useMemo, useRef, useState } from 'react';
import ReactMarkdown from 'react-markdown';

const API_BASE = 'http://localhost:8000/api';

const defaultSystemPrompt =
  'You are a helpful assistant. Use clear markdown headings, lists, and fenced code blocks when useful.';

function parseSSEStream(response, handlers) {
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  const pump = () =>
    reader.read().then(({ done, value }) => {
      if (done) {
        handlers.onClose && handlers.onClose();
        return;
      }
      buffer += decoder.decode(value, { stream: true });
      let sep;
      while ((sep = buffer.indexOf('\n\n')) !== -1) {
        const frame = buffer.slice(0, sep);
        buffer = buffer.slice(sep + 2);
        let event = 'message';
        const dataLines = [];
        for (const line of frame.split('\n')) {
          if (line.startsWith('event: ')) event = line.slice(7).trim();
          else if (line.startsWith('data: ')) dataLines.push(line.slice(6));
        }
        if (dataLines.length === 0) continue;
        let data;
        try {
          data = JSON.parse(dataLines.join('\n'));
        } catch (e) {
          data = dataLines.join('\n');
        }
        handlers.onEvent && handlers.onEvent(event, data);
      }
      return pump();
    });
  return pump();
}

export default function App() {
  const [conversationId, setConversationId] = useState(null);
  const [systemPrompt, setSystemPrompt] = useState(defaultSystemPrompt);
  const [userPrompt, setUserPrompt] = useState('');
  const [messages, setMessages] = useState([]);
  const [mode, setMode] = useState('chat');
  const [docId, setDocId] = useState('');
  const [docStatus, setDocStatus] = useState('');
  const [isSending, setIsSending] = useState(false);
  const [toolStatus, setToolStatus] = useState('');
  const chatEndRef = useRef(null);

  useEffect(() => {
    const stored = localStorage.getItem('conversationId');
    if (stored) {
      setConversationId(stored);
      loadConversation(stored);
    }
  }, []);

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth', block: 'end' });
  }, [messages, toolStatus]);

  const canSend = useMemo(() => {
    if (!userPrompt.trim() || isSending) return false;
    if (mode === 'rag' && !docId.trim()) return false;
    return true;
  }, [userPrompt, mode, docId, isSending]);

  async function loadConversation(id) {
    try {
      const response = await fetch(`${API_BASE}/conversations/${id}`);
      if (!response.ok) return;
      const data = await response.json();
      setMessages(
        (data.messages || []).map((m) => ({ role: m.role, content: m.content, sources: null }))
      );
    } catch (err) {
      console.error(err);
    }
  }

  async function startNewConversation() {
    try {
      const response = await fetch(`${API_BASE}/conversations`, { method: 'POST' });
      const data = await response.json();
      localStorage.setItem('conversationId', data.conversation_id);
      setConversationId(data.conversation_id);
      setMessages([]);
      setToolStatus('');
    } catch (err) {
      console.error(err);
    }
  }

  async function handleUpload(event) {
    const file = event.target.files?.[0];
    if (!file) return;
    setDocStatus('Uploading and extracting (this may take a minute)...');
    const formData = new FormData();
    formData.append('file', file);
    try {
      const response = await fetch(`${API_BASE}/documents`, {
        method: 'POST',
        body: formData,
      });
      if (!response.ok) throw new Error(await response.text());
      const data = await response.json();
      setDocId(data.doc_id);
      setDocStatus(
        `Loaded ${data.filename} — ${data.page_count} pages, ${data.section_count} sections`
      );
      setMode('rag');
    } catch (err) {
      console.error(err);
      setDocStatus('Upload failed. Check backend logs.');
    }
  }

  async function sendMessage() {
    if (!canSend) return;
    const nextUser = userPrompt.trim();
    setIsSending(true);
    setToolStatus('');
    setUserPrompt('');
    setMessages((prev) => [
      ...prev,
      { role: 'user', content: nextUser, sources: null },
      { role: 'assistant', content: '', sources: null, streaming: true },
    ]);

    const payload = {
      conversation_id: conversationId,
      system_prompt: systemPrompt,
      user_prompt: nextUser,
      mode,
      doc_id: mode === 'rag' ? docId : null,
    };

    let collectedSources = null;
    try {
      const response = await fetch(`${API_BASE}/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });
      if (!response.ok) throw new Error(await response.text());

      await parseSSEStream(response, {
        onEvent: (event, data) => {
          if (event === 'start') {
            if (data?.conversation_id && !conversationId) {
              localStorage.setItem('conversationId', data.conversation_id);
              setConversationId(data.conversation_id);
            }
          } else if (event === 'token') {
            const chunk = typeof data === 'string' ? data : data?.data || '';
            setMessages((prev) => {
              const copy = [...prev];
              const last = copy[copy.length - 1];
              if (last && last.role === 'assistant') {
                copy[copy.length - 1] = { ...last, content: last.content + chunk };
              }
              return copy;
            });
          } else if (event === 'status') {
            setToolStatus(data?.message || '');
          } else if (event === 'sources') {
            collectedSources = data;
          } else if (event === 'error') {
            setMessages((prev) => {
              const copy = [...prev];
              const last = copy[copy.length - 1];
              if (last && last.role === 'assistant') {
                copy[copy.length - 1] = {
                  ...last,
                  content: last.content + `\n\n_Error: ${data?.message || 'unknown'}_`,
                };
              }
              return copy;
            });
          } else if (event === 'done') {
            setMessages((prev) => {
              const copy = [...prev];
              const last = copy[copy.length - 1];
              if (last && last.role === 'assistant') {
                copy[copy.length - 1] = {
                  ...last,
                  streaming: false,
                  sources: collectedSources,
                };
              }
              return copy;
            });
            setToolStatus('');
          }
        },
      });
    } catch (err) {
      console.error(err);
      setMessages((prev) => {
        const copy = [...prev];
        const last = copy[copy.length - 1];
        if (last && last.role === 'assistant' && !last.content) {
          copy[copy.length - 1] = {
            ...last,
            content: 'Sorry, something went wrong. Check backend logs.',
            streaming: false,
          };
        }
        return copy;
      });
    } finally {
      setIsSending(false);
    }
  }

  return (
    <div className="app">
      <header className="hero">
        <div>
          <p className="eyebrow">Gemma 4 E2B POC</p>
          <h1>PageIndex RAG + long-term chat</h1>
          <p className="subhead">
            FastAPI + llama-cpp-python, OpenDataLoader PDF ingestion, PageIndex tool-call RAG, markdown output.
          </p>
        </div>
        <div className="controls">
          <label className="field">
            <span>System Prompt</span>
            <textarea
              value={systemPrompt}
              onChange={(e) => setSystemPrompt(e.target.value)}
              rows={4}
            />
          </label>
          <label className="field">
            <span>Mode</span>
            <div className="pill-row">
              <button
                className={mode === 'chat' ? 'pill active' : 'pill'}
                onClick={() => setMode('chat')}
                type="button"
              >
                General Chat
              </button>
              <button
                className={mode === 'rag' ? 'pill active' : 'pill'}
                onClick={() => setMode('rag')}
                type="button"
                disabled={!docId}
                title={!docId ? 'Upload a PDF first' : ''}
              >
                RAG: PDF
              </button>
            </div>
          </label>
          <div className="field">
            <span>PDF</span>
            <input type="file" accept="application/pdf" onChange={handleUpload} />
            <div className="status">{docStatus || 'No document loaded.'}</div>
          </div>
          <div className="field">
            <button type="button" onClick={startNewConversation}>
              New conversation
            </button>
          </div>
        </div>
      </header>

      <main className="chat">
        <section className="chat-panel">
          <div className="chat-window">
            {messages.length === 0 && (
              <div className="empty">Start chatting to see markdown-rendered responses.</div>
            )}
            {messages.map((msg, idx) => (
              <div key={idx} className={`bubble ${msg.role}`}>
                <div className="role">{msg.role}</div>
                <ReactMarkdown>
                  {msg.content || (msg.streaming ? '…' : '')}
                </ReactMarkdown>
                {msg.sources && msg.sources.length > 0 && (
                  <div className="sources">
                    <span className="sources-label">Sources:</span>
                    {msg.sources.map((s, i) => (
                      <span className="source-chip" key={i}>
                        {s.section_id ? `[${s.section_id}] ${s.title}` : s.title}
                        {s.pages && s.pages.length > 0 && ` (p.${s.pages.join(', ')})`}
                      </span>
                    ))}
                  </div>
                )}
              </div>
            ))}
            {toolStatus && (
              <div className="bubble assistant tool-status">
                <div className="role">thinking</div>
                <em>{toolStatus}</em>
              </div>
            )}
            <div ref={chatEndRef} />
          </div>
          <div className="composer">
            <textarea
              placeholder={mode === 'rag' ? 'Ask about the uploaded PDF...' : 'Ask anything...'}
              value={userPrompt}
              onChange={(e) => setUserPrompt(e.target.value)}
              rows={3}
              onKeyDown={(e) => {
                if (e.key === 'Enter' && (e.metaKey || e.ctrlKey)) {
                  e.preventDefault();
                  sendMessage();
                }
              }}
            />
            <button type="button" onClick={sendMessage} disabled={!canSend}>
              {isSending ? 'Thinking…' : 'Send'}
            </button>
          </div>
        </section>
      </main>
    </div>
  );
}
