import React, { useState, useRef, useEffect } from "react";
import "./App.css";

// Single-file React component (default export). Tailwind CSS classes are used for styling.
// Usage: render <App /> in your React application. Requires Tailwind configured.

export default function App() {
  const [userId, setUserId] = useState(() => {
    try {
      const stored = localStorage.getItem("userId");
      if (stored) return stored;
    } catch (e) {}
    const id = `user-${Math.floor(Math.random() * 10000)}`;
    try {
      localStorage.setItem("userId", id);
    } catch (e) {}
    return id;
  });
  const [messages, setMessages] = useState([]); // { id, role: 'user'|'assistant'|'system', text }
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [recording, setRecording] = useState(false);
  const mediaRecorderRef = useRef(null);
  const recordedChunksRef = useRef([]);
  const audioRef = useRef(null);

  // Helper to append a message
  const pushMessage = (role, text) => {
    setMessages((m) => [...m, { id: `${Date.now()}-${Math.random()}`, role, text }]);
  };

  // Send plain text to /api/chat
  const sendText = async () => {
    const text = (input || "").trim();
    if (!text) return;
    pushMessage("user", text);
    setInput("");
    setLoading(true);
    try {
      const res = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ user_id: userId, message: text, save_memory: true }),
      });
      const data = await res.json();
      const reply = data.reply || data;
      pushMessage("assistant", reply);
    } catch (err) {
      console.error(err);
      pushMessage("assistant", "Error: could not reach server.");
    } finally {
      setLoading(false);
    }
  };

  // Start recording with MediaRecorder
  const startRecording = async () => {
    recordedChunksRef.current = [];
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const mr = new MediaRecorder(stream);
      mediaRecorderRef.current = mr;

      mr.ondataavailable = (e) => {
        if (e.data && e.data.size > 0) recordedChunksRef.current.push(e.data);
      };

      mr.onstop = () => {
        // stop all tracks
        stream.getTracks().forEach((t) => t.stop());
      };

      mr.start();
      setRecording(true);
    } catch (e) {
      console.error("microphone error", e);
      alert("Could not access microphone. Allow microphone permission and try again.");
    }
  };

  // Stop recording and upload blob
  const stopRecording = async ({ runChat=true, saveMemory=true } = {}) => {
    const mr = mediaRecorderRef.current;
    if (!mr) return;
    setRecording(false);
    try {
      mr.stop();
    } catch (e) {
      console.warn(e);
    }

    const blob = new Blob(recordedChunksRef.current, { type: "audio/webm" });
    // play preview
    try {
      if (audioRef.current) audioRef.current.src = URL.createObjectURL(blob);
    } catch (e) {}

    // upload to /api/voice/transcribe?user_id=...&save_memory=...&run_chat=...
    setLoading(true);
    try {
      const fd = new FormData();
      const filename = `rec-${Date.now()}.webm`;
      // Convert blob to File for server compatibility
      const file = new File([blob], filename, { type: blob.type });
      fd.append("file", file);
      const params = new URLSearchParams({ user_id: userId, save_memory: String(saveMemory), run_chat: String(runChat) });
      const res = await fetch(`/api/voice/transcribe?${params.toString()}`, { method: "POST", body: fd });
      const data = await res.json();

      if (data.error) {
        pushMessage("assistant", `Transcription error: ${data.error}`);
      } else {
        const transcription = data.transcription || "(no transcription)";
        pushMessage("user", transcription);
        if (data.reply) pushMessage("assistant", data.reply);
        // optionally show used memories in console
        if (data.used_memories) console.log("Used memories:", data.used_memories);
      }
    } catch (e) {
      console.error(e);
      pushMessage("assistant", "Error uploading audio or getting response.");
    } finally {
      setLoading(false);
    }
  };

  // Simple keyboard send
  const onKeyDown = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendText();
    }
  };

  // Small helper to clear memory (calls /api/memory/clear)
  const clearMemories = async () => {
    if (!confirm("Clear all memories? This cannot be undone.")) return;
    try {
      // pass user_id so the backend clears the current user's memories
      const res = await fetch(`/api/memory/clear?user_id=${encodeURIComponent(userId)}`, { method: "DELETE" });
      const j = await res.json();
      if (j.status === "ok") {
        // clear local UI messages so user sees immediate update
        setMessages([]);
        alert("Memories cleared.");
      } else {
        console.warn('clear response', j);
        alert("Clearing memories returned unexpected response.");
      }
    } catch (e) {
      console.error(e);
      alert("Failed to clear memories.");
    }
  };

  return (
    <div className="min-h-screen bg-gray-50 grid place-items-center p-6">
      <div className="w-full max-w-3xl bg-white rounded-2xl shadow-md p-4 grid grid-rows-[auto,1fr,auto] gap-4 mx-auto">
        {/* Header */}
        <header className="flex items-center justify-between">
          <div>
            <h1 className="text-2xl font-semibold">Memory Agent — Voice + Chat</h1>
            <p className="text-sm text-gray-500">User: <span className="font-mono">{userId}</span></p>
          </div>
          <div className="flex items-center gap-2">
            <button
              onClick={() => navigator.clipboard.writeText(userId)}
              title="Copy ID"
              className="p-2 rounded-md border text-sm"
              aria-label="Copy ID"
            >
              {/* clipboard icon */}
              <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12H7a2 2 0 01-2-2V7a2 2 0 012-2h6l2 2h4a2 2 0 012 2v6a2 2 0 01-2 2h-2" />
                <rect x="9" y="9" width="6" height="10" rx="2" ry="2" />
              </svg>
            </button>
            <button
              onClick={clearMemories}
              title="Clear Memories"
              className="p-2 rounded-md bg-red-50 text-red-700 border"
              aria-label="Clear Memories"
            >
              {/* trash icon */}
              <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6M1 7h22M10 3h4l1 2H9l1-2z" />
              </svg>
            </button>
          </div>
        </header>

        {/* Chat area */}
        <main className="overflow-auto p-4 border rounded-lg" style={{ minHeight: 320 }}>
          <div className="flex flex-col gap-3">
            {messages.length === 0 && (
              <div className="text-center text-gray-400">No messages yet — record audio or type below.</div>
            )}
            {messages.map((m) => (
              <div key={m.id} className={`max-w-[80%] ${m.role === 'user' ? 'self-end bg-indigo-50 text-indigo-900' : 'self-start bg-gray-100 text-gray-900'} p-3 rounded-xl`}> 
                <div className="whitespace-pre-wrap">{m.text}</div>
                <div className="text-xs text-gray-400 mt-1">{m.role}</div>
              </div>
            ))}
          </div>
        </main>

        {/* Controls */}
        <footer className="flex flex-col gap-3">
            <div className="flex gap-2 items-center">
            <textarea
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={onKeyDown}
              placeholder="Type a message and press Enter to send"
              className="flex-1 border rounded-md p-2 resize-none h-20"
            />
            <div className="flex flex-col gap-2">
                <button onClick={sendText} disabled={loading} className="px-3 py-2 rounded-md bg-indigo-600 text-white flex items-center justify-center" title="Send" aria-label="Send">
                  {/* send / paper plane icon */}
                  <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" viewBox="0 0 20 20" fill="currentColor">
                    <path d="M2.94 2.94a1.5 1.5 0 012.12 0L17 14.88V17a1 1 0 01-1 1h-2.12L2.94 5.06a1.5 1.5 0 010-2.12z" />
                  </svg>
                </button>
                <button
                  onClick={() => { if (!recording) startRecording(); else stopRecording(); }}
                  className={`px-3 py-2 rounded-md flex items-center justify-center ${recording ? 'bg-red-500 text-white' : 'bg-green-500 text-white'}`}
                  title={recording ? 'Stop recording' : 'Record'}
                  aria-label={recording ? 'Stop recording' : 'Record'}
                >
                  {recording ? (
                    // stop icon
                    <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" viewBox="0 0 20 20" fill="currentColor">
                      <rect x="4" y="4" width="12" height="12" rx="2" />
                    </svg>
                  ) : (
                    // mic icon
                    <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" viewBox="0 0 20 20" fill="currentColor">
                      <path d="M10 12a3 3 0 003-3V5a3 3 0 10-6 0v4a3 3 0 003 3z" />
                      <path d="M5 10a5 5 0 0010 0h-2a3 3 0 01-6 0H5z" />
                      <path d="M9 17h2v3H9z" />
                    </svg>
                  )}
                </button>
            </div>
          </div>

          <div className="flex items-center justify-between text-sm text-gray-500">
            <div>Tip: record voice and the backend will transcribe + ask the model.</div>
            <div>{loading ? 'Processing…' : 'Idle'}</div>
          </div>

          {/* hidden audio preview for recorded blob */}
          <audio ref={audioRef} controls className="hidden" />
        </footer>
      </div>
    </div>
  );
}
