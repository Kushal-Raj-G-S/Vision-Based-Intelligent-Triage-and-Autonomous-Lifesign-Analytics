'use client';

import React, { useState, useEffect, useRef } from 'react';

// TypeScript Interfaces
interface ChatMessage {
  role: 'user' | 'model';
  content: string;
  image?: string; // Base64 encoding of clinical image
}

interface PatientRecord {
  id: string;
  name: string;
  video_path: string;
  timestamp: string;
  esi_level: number;
  priority_score: number;
  primary_diagnosis: string;
  is_shock: boolean;
  triage_summary: string;
  agent_output: string;
}

interface CameraDevice {
  index: number;
  label: string;
}

interface Metrics {
  bpm: number;
  confidence: number;
  status: string;
  snr_db: number;
  sqi: number;
  classification: string;
  ohi: number;
  stability: number;
  stability_indicator: string;
  rr: number;
  rr_confidence: number;
  rr_classification: string;
  hrv: number;
  stress_index: number;
  warnings: string[];
  remark: string;
  estimated_lux: number;
  motion_delta: number;
  is_live: boolean;
  calibration_done: boolean;
  ppg_signal: number[];
  calibration_progress: number;
  face_detected?: boolean;
}

const BACKEND_URL = "http://127.0.0.1:5002";

type NavigationTab = 'monitor' | 'queue' | 'crew' | 'chat';

export default function Home() {
  // Navigation
  const [activeTab, setActiveTab] = useState<NavigationTab>('monitor');

  // Connection and settings state
  const [backendOnline, setBackendOnline] = useState<boolean>(false);
  const [cameras, setCameras] = useState<CameraDevice[]>([]);
  const [selectedCamera, setSelectedCamera] = useState<number | null>(null);
  
  // Real-time metric states
  const [metrics, setMetrics] = useState<Metrics>({
    bpm: 0, confidence: 0, status: 'DISCONNECTED',
    snr_db: 0, sqi: 0, classification: 'UNKNOWN',
    ohi: 0, stability: 0, stability_indicator: '--',
    rr: 0, rr_confidence: 0, rr_classification: '--',
    hrv: 0, stress_index: 0, warnings: [],
    remark: '', estimated_lux: 0, motion_delta: 0,
    is_live: false, calibration_done: false, ppg_signal: [],
    calibration_progress: 0, face_detected: false
  });
  
  // Triage Queue & Multi-Agent Triage state
  const [triageQueue, setTriageQueue] = useState<PatientRecord[]>([]);
  const [isTriageRunning, setIsTriageRunning] = useState<boolean>(false);
  const [lastTriageResult, setLastTriageResult] = useState<PatientRecord | null>(null);
  const [activeAgentTab, setActiveAgentTab] = useState<'perception' | 'diagnostic' | 'coordinator'>('perception');
  
  // Chat state
  const [chatHistory, setChatHistory] = useState<ChatMessage[]>([]);
  const [chatInput, setChatInput] = useState<string>('');
  const [selectedImage, setSelectedImage] = useState<string | null>(null);
  const [isChatLoading, setIsChatLoading] = useState<boolean>(false);
  const [isRecording, setIsRecording] = useState<boolean>(false);
  
  // File upload state
  const [uploadMessage, setUploadMessage] = useState<{ text: string; type: 'success' | 'error' | '' }>({ text: '', type: '' });
  // Video feed reconnection key — incrementing forces img src reload
  const [videoFeedKey, setVideoFeedKey] = useState<number>(0);
  // 30-second session timer and auto-report
  const [sessionTimer, setSessionTimer] = useState<number>(30);
  const [sessionReport, setSessionReport] = useState<any>(null);
  const [reportLoading, setReportLoading] = useState<boolean>(false);
  const timerRef = useRef<NodeJS.Timeout | null>(null);
  const reportFetchedRef = useRef<boolean>(false);
  
  const [theme, setTheme] = useState<'dark' | 'light'>('dark');
  const [patientName, setPatientName] = useState<string>('');
  
  // Refs
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const recognitionRef = useRef<any>(null);
  const chatEndRef = useRef<HTMLDivElement | null>(null);
  const imageInputRef = useRef<HTMLInputElement | null>(null);
  const sessionStartedRef = useRef<boolean>(false); // Prevent double auto-start
  
  // Polling intervals
  const statusPollInterval = useRef<NodeJS.Timeout | null>(null);

  // 1. Verify Backend Online and fetch cameras & queue
  const checkBackendStatus = async () => {
    try {
      const res = await fetch(`${BACKEND_URL}/api/cameras`);
      if (res.ok) {
        setBackendOnline(true);
        const data = await res.json();
        setCameras(data.cameras || []);
        if (data.default !== null && data.default !== undefined) {
          setSelectedCamera((prev) => prev !== null ? prev : data.default);
        }
        fetchQueue();
      } else {
        setBackendOnline(false);
        sessionStartedRef.current = false; // backend went offline, allow re-start
      }
    } catch (e) {
      setBackendOnline(false);
      sessionStartedRef.current = false;
    }
  };

  const fetchQueue = async () => {
    try {
      const res = await fetch(`${BACKEND_URL}/api/triage_queue`);
      if (res.ok) {
        const data = await res.json();
        setTriageQueue(data.queue || []);
      }
    } catch (e) {
      console.error("Error fetching triage queue:", e);
    }
  };

  useEffect(() => {
    checkBackendStatus();
    const interval = setInterval(checkBackendStatus, 5000);
    return () => clearInterval(interval);
  }, []);

  // Set up status polling
  useEffect(() => {
    if (backendOnline) {
      statusPollInterval.current = setInterval(async () => {
        try {
          const res = await fetch(`${BACKEND_URL}/status`);
          if (res.ok) {
            const data = await res.json();
            setMetrics(data);
          }
        } catch (e) {
          console.error("Status poll error:", e);
        }
      }, 300);
    } else {
      if (statusPollInterval.current) clearInterval(statusPollInterval.current);
    }
    return () => {
      if (statusPollInterval.current) clearInterval(statusPollInterval.current);
    };
  }, [backendOnline]);

  // Scroll to bottom of chat
  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [chatHistory, isChatLoading, activeTab]);

  // 30-second countdown timer — starts when calibration_done flips true
  useEffect(() => {
    if (metrics.calibration_done && metrics.is_live) {
      // Start countdown if not already running
      if (timerRef.current === null && sessionTimer <= 30 && !reportFetchedRef.current) {
        timerRef.current = setInterval(() => {
          setSessionTimer(prev => {
            if (prev <= 1) {
              // Time up — fetch report
              clearInterval(timerRef.current!);
              timerRef.current = null;
              if (!reportFetchedRef.current) {
                reportFetchedRef.current = true;
                setReportLoading(true);
                fetch(`${BACKEND_URL}/api/generate_report`, { method: 'POST' })
                  .then(r => r.json())
                  .then(d => { 
                    if (d.success) {
                      setSessionReport(d.report);
                      // Auto-run Clinical Crew triage with custom patient name
                      handleRunTriage(patientName);
                    }
                  })
                  .catch(console.error)
                  .finally(() => setReportLoading(false));
              }
              return 0;
            }
            return prev - 1;
          });
        }, 1000);
      }
    } else {
      // Reset on disconnect / new session
      if (timerRef.current) { clearInterval(timerRef.current); timerRef.current = null; }
      if (!metrics.calibration_done) {
        setSessionTimer(30);
        setSessionReport(null);
        reportFetchedRef.current = false;
      }
    }
    return () => {};
  }, [metrics.calibration_done, metrics.is_live, sessionReport]);

  // 2. Draw rPPG wave signal on HTML5 Canvas
  useEffect(() => {
    if (activeTab !== 'monitor') return; // Draw only when visible
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    // Draw background grid
    ctx.strokeStyle = 'rgba(6, 182, 212, 0.04)';
    ctx.lineWidth = 1;
    for (let i = 0; i < canvas.width; i += 40) {
      ctx.beginPath();
      ctx.moveTo(i, 0);
      ctx.lineTo(i, canvas.height);
      ctx.stroke();
    }
    for (let i = 0; i < canvas.height; i += 30) {
      ctx.beginPath();
      ctx.moveTo(0, i);
      ctx.lineTo(canvas.width, i);
      ctx.stroke();
    }

    const signal = metrics.ppg_signal || [];
    if (signal.length === 0) {
      // Draw idle scanning line
      ctx.strokeStyle = 'rgba(6, 182, 212, 0.2)';
      ctx.lineWidth = 2.5;
      ctx.beginPath();
      ctx.moveTo(0, canvas.height / 2);
      ctx.lineTo(canvas.width, canvas.height / 2);
      ctx.stroke();
      return;
    }

    // Smooth signal mapping
    const maxVal = Math.max(...signal);
    const minVal = Math.min(...signal);
    const range = maxVal - minVal || 1;

    // Shaded gradient area under the curve
    ctx.beginPath();
    ctx.moveTo(0, canvas.height);
    for (let i = 0; i < signal.length; i++) {
      const x = (i / (signal.length - 1)) * canvas.width;
      const y = canvas.height - 20 - ((signal[i] - minVal) / range) * (canvas.height - 40);
      ctx.lineTo(x, y);
    }
    ctx.lineTo(canvas.width, canvas.height);
    ctx.closePath();
    
    const grad = ctx.createLinearGradient(0, 0, 0, canvas.height);
    grad.addColorStop(0, 'rgba(6, 182, 212, 0.20)');
    grad.addColorStop(1, 'rgba(6, 182, 212, 0.00)');
    ctx.fillStyle = grad;
    ctx.fill();

    // Draw the neon stroke line
    ctx.beginPath();
    ctx.strokeStyle = 'rgba(6, 182, 212, 0.95)';
    ctx.lineWidth = 3;
    ctx.shadowBlur = 12;
    ctx.shadowColor = 'rgba(6, 182, 212, 0.7)';

    for (let i = 0; i < signal.length; i++) {
      const x = (i / (signal.length - 1)) * canvas.width;
      const y = canvas.height - 20 - ((signal[i] - minVal) / range) * (canvas.height - 40);
      if (i === 0) {
        ctx.moveTo(x, y);
      } else {
        ctx.lineTo(x, y);
      }
    }
    ctx.stroke();
    ctx.shadowBlur = 0; // reset
  }, [metrics.ppg_signal, metrics.status, activeTab]);

  // 3. Audio Recording STT / Voice Input
  useEffect(() => {
    if (typeof window !== 'undefined') {
      const SpeechRecognition = (window as any).SpeechRecognition || (window as any).webkitSpeechRecognition;
      if (SpeechRecognition) {
        const rec = new SpeechRecognition();
        rec.continuous = false;
        rec.interimResults = false;
        rec.lang = 'en-US';

        rec.onstart = () => setIsRecording(true);
        rec.onend = () => setIsRecording(false);
        rec.onerror = () => setIsRecording(false);
        rec.onresult = (event: any) => {
          const resultText = event.results[0][0].transcript;
          setChatInput(resultText);
        };
        recognitionRef.current = rec;
      }
    }
  }, []);

  const toggleRecording = () => {
    if (!recognitionRef.current) {
      alert("Speech Recognition API is not supported in this browser. Please type your query.");
      return;
    }
    if (isRecording) {
      recognitionRef.current.stop();
    } else {
      recognitionRef.current.start();
    }
  };

  // 4. Voice TTS output
  const speakText = (text: string) => {
    if (typeof window !== 'undefined' && window.speechSynthesis) {
      window.speechSynthesis.cancel();
      const cleanMsg = text.replace(/^\[[a-z]{2}-[A-Z]{2}\]\s*/i, '');
      const utterance = new SpeechSynthesisUtterance(cleanMsg);
      window.speechSynthesis.speak(utterance);
    }
  };

  // 5. Handlers
  const handleStartWebcam = async () => {
    setUploadMessage({ text: '', type: '' });
    try {
      const res = await fetch(`${BACKEND_URL}/start_webcam`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ source: selectedCamera })
      });
      const data = await res.json();
      if (data.success) {
        setUploadMessage({ text: `Webcam session initialized.`, type: 'success' });
        // Force video feed img to reconnect (backend reset kills old MJPEG stream)
        setTimeout(() => setVideoFeedKey(k => k + 1), 400);
      } else {
        setUploadMessage({ text: data.error || 'Failed to start webcam.', type: 'error' });
      }
    } catch (e) {
      setUploadMessage({ text: 'Error starting webcam.', type: 'error' });
    }
  };

  const handleReleaseCamera = async () => {
    try {
      await fetch(`${BACKEND_URL}/release_camera`, { method: 'POST' });
      setUploadMessage({ text: 'Optical hardware released.', type: 'success' });
    } catch (e) {
      console.error(e);
    }
  };

  const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (!files || files.length === 0) return;
    const file = files[0];

    const formData = new FormData();
    formData.append('video', file);

    setUploadMessage({ text: 'Uploading intake media record...', type: 'success' });

    try {
      const res = await fetch(`${BACKEND_URL}/upload`, {
        method: 'POST',
        body: formData
      });
      const data = await res.json();
      if (data.success) {
        setUploadMessage({ text: `Media loaded: ${data.message}`, type: 'success' });
      } else {
        setUploadMessage({ text: data.error || 'Upload failed.', type: 'error' });
      }
    } catch (err) {
      setUploadMessage({ text: 'Network error uploading file.', type: 'error' });
    }
  };

  const handleImageAttachment = (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (!files || files.length === 0) return;
    const file = files[0];
    
    const reader = new FileReader();
    reader.onloadend = () => {
      setSelectedImage(reader.result as string);
    };
    reader.readAsDataURL(file);
  };

  const handleRunTriage = async (customName?: string) => {
    setIsTriageRunning(true);
    setLastTriageResult(null);
    try {
      const isLive = metrics.is_live || (metrics.status === 'OK' || metrics.status === 'CALIBRATING');
      const bodySource = isLive ? 'live' : '';
      const nameToSubmit = customName || patientName || '';
      
      let res;
      if (bodySource) {
        res = await fetch(`${BACKEND_URL}/api/triage_run`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ source: 'live', patient_name: nameToSubmit })
        });
      } else {
        const formData = new FormData();
        formData.append('patient_name', nameToSubmit);
        res = await fetch(`${BACKEND_URL}/api/triage_run`, {
          method: 'POST',
          body: formData
        });
      }

      if (res.ok) {
        const data = await res.json();
        if (data.success) {
          setLastTriageResult(data.patient_record);
          fetchQueue();
          setActiveAgentTab('coordinator');
        } else {
          alert(`Triage Crew failed: ${data.error}`);
        }
      } else {
        alert("Server error running Triage Crew.");
      }
    } catch (e) {
      alert(`Network error running Triage Crew: ${e}`);
    } finally {
      setIsTriageRunning(false);
    }
  };

  const handleClearQueue = async () => {
    if (!confirm("Are you sure you want to clear the entire triage queue?")) return;
    try {
      const res = await fetch(`${BACKEND_URL}/api/triage_queue`, { method: 'DELETE' });
      if (res.ok) {
        fetchQueue();
        setLastTriageResult(null);
      }
    } catch (e) {
      console.error(e);
    }
  };

  const handleSendChatMessage = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!chatInput.trim() && !selectedImage) return;

    const userMsg: ChatMessage = { 
      role: 'user', 
      content: chatInput,
      image: selectedImage || undefined
    };
    
    const updatedHistory = [...chatHistory, userMsg];
    setChatHistory(updatedHistory);
    setChatInput('');
    setSelectedImage(null);
    setIsChatLoading(true);

    try {
      const res = await fetch(`${BACKEND_URL}/api/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ 
          message: userMsg.content, 
          history: chatHistory.map(m => ({ role: m.role, content: m.content })),
          image: userMsg.image 
        })
      });
      const data = await res.json();
      if (data.response) {
        const modelMsg: ChatMessage = { role: 'model', content: data.response };
        setChatHistory([...updatedHistory, modelMsg]);
        speakText(data.response);
      } else {
        setChatHistory([...updatedHistory, { role: 'model', content: `Error: ${data.error || 'No response.'}` }]);
      }
    } catch (e) {
      setChatHistory([...updatedHistory, { role: 'model', content: 'Connection error connecting to ARIA.' }]);
    } finally {
      setIsChatLoading(false);
    }
  };

  // UI Helpers
  const getEsiClass = (level: number) => {
    switch (level) {
      case 1: return 'bg-red-500/10 border-red-500 text-red-400 shadow-[0_0_15px_rgba(239,68,68,0.15)] animate-pulse';
      case 2: return 'bg-orange-500/10 border-orange-500 text-orange-400 shadow-[0_0_15px_rgba(245,158,11,0.1)]';
      case 3: return 'bg-yellow-500/10 border-yellow-500 text-yellow-400';
      case 4: return 'bg-emerald-500/10 border-emerald-500 text-emerald-400';
      case 5: return 'bg-cyan-500/10 border-cyan-500 text-cyan-400';
      default: return 'bg-zinc-800 border-zinc-700 text-zinc-300';
    }
  };

  const getHeartbeatDuration = () => {
    const rate = metrics.bpm > 0 ? metrics.bpm : 72;
    return `${60 / rate}s`;
  };

  return (
    <div className={`flex-1 font-sans min-h-screen pb-12 antialiased selection:bg-cyan-500 selection:text-black transition-colors duration-200 ${
      theme === 'light' ? 'light-theme bg-slate-50 text-slate-900' : 'bg-[#030712] text-zinc-100'
    }`}>
      
      {/* Keyframe Injection for custom scanlines, glows, and animations */}
      <style jsx global>{`
        @keyframes scan {
          0% { transform: translateY(-100%); }
          100% { transform: translateY(100%); }
        }
        .animate-scan {
          animation: scan 6s linear infinite;
        }
        .shadow-glow-cyan {
          box-shadow: 0 0 20px rgba(6, 182, 212, 0.2);
        }
        .shadow-glow-red {
          box-shadow: 0 0 20px rgba(239, 68, 68, 0.25);
        }

        /* ── LIGHT THEME STYLES ── */
        .light-theme {
          background-color: #f8fafc !important;
          color: #0f172a !important;
        }
        .light-theme header {
          background-color: rgba(255, 255, 255, 0.9) !important;
          border-color: #e2e8f0 !important;
          box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -2px rgba(0, 0, 0, 0.05) !important;
        }
        .light-theme header h1, .light-theme header span {
          color: #0f172a !important;
        }
        .light-theme header p {
          color: #64748b !important;
        }
        /* Top navigator */
        .light-theme nav > div {
          background-color: rgba(255, 255, 255, 0.95) !important;
          border-color: #e2e8f0 !important;
          box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05) !important;
        }
        .light-theme nav button {
          color: #64748b !important;
        }
        .light-theme nav button:hover {
          background-color: #f1f5f9 !important;
          color: #0f172a !important;
        }
        /* Cards */
        .light-theme .bg-\[\#090e1e\]\/60,
        .light-theme .bg-\[\#090e1e\]\/80 {
          background-color: rgba(255, 255, 255, 0.9) !important;
          border-color: #e2e8f0 !important;
          box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.05), 0 4px 6px -4px rgba(0, 0, 0, 0.05) !important;
        }
        /* Internal blocks (black backgrounds) */
        .light-theme .bg-\[\#02040a\],
        .light-theme .bg-\[\#02040a\]\/40,
        .light-theme .bg-\[\#02040a\]\/60,
        .light-theme .bg-\[\#02040a\]f0,
        .light-theme .bg-\[\#070b19\] {
          background-color: #f1f5f9 !important;
          border-color: #e2e8f0 !important;
        }
        /* Border overrides */
        .light-theme .border-zinc-800,
        .light-theme .border-zinc-800\/80,
        .light-theme .border-zinc-850,
        .light-theme .border-zinc-800\/60,
        .light-theme .border-zinc-700\/80,
        .light-theme .border-indigo-800\/40 {
          border-color: #e2e8f0 !important;
        }
        /* Text color overrides */
        .light-theme .text-white,
        .light-theme .text-zinc-100,
        .light-theme .text-zinc-150,
        .light-theme .text-zinc-200,
        .light-theme .text-zinc-300 {
          color: #0f172a !important;
        }
        .light-theme .text-zinc-450,
        .light-theme .text-zinc-500,
        .light-theme .text-zinc-550 {
          color: #64748b !important;
        }
        .light-theme .text-zinc-400 {
          color: #475569 !important;
        }
        /* Live Readings items styling */
        .light-theme .bg-\[\#070b19\] {
          background-color: #ffffff !important;
          border-color: #e2e8f0 !important;
        }
        /* Forms, inputs & controls */
        .light-theme select,
        .light-theme input {
          background-color: #ffffff !important;
          border-color: #cbd5e1 !important;
          color: #0f172a !important;
        }
        .light-theme select:focus,
        .light-theme input:focus {
          border-color: #06b6d4 !important;
        }
        /* Interactive controls panel bg */
        .light-theme button.bg-zinc-900 {
          background-color: #e2e8f0 !important;
          color: #0f172a !important;
          border-color: #cbd5e1 !important;
        }
        .light-theme button.bg-zinc-900:hover {
          background-color: #cbd5e1 !important;
        }
        /* Scrollbars and status displays */
        .light-theme .bg-\[\#0b1022\] {
          background-color: #e2e8f0 !important;
          border-color: #cbd5e1 !important;
          color: #0f172a !important;
        }
        .light-theme .bg-\[\#0b1022\] .text-zinc-500 {
          color: #475569 !important;
        }
        /* Chat view message bubbles */
        .light-theme .bg-zinc-900\/60 {
          background-color: #e2e8f0 !important;
          color: #0f172a !important;
          border-color: #cbd5e1 !important;
        }
        .light-theme .text-zinc-250 {
          color: #1e293b !important;
        }
      `}</style>

      {/* 1. Main Header */}
      <header className="sticky top-0 z-50 bg-[#070b19]/90 backdrop-blur-xl border-b border-zinc-800/80 px-6 py-4 flex flex-wrap items-center justify-between gap-4 shadow-xl">
        <div className="flex items-center gap-3">
          <div className="w-12 h-12 rounded-xl bg-zinc-950 flex items-center justify-center shadow-lg shadow-cyan-500/20 relative overflow-hidden border border-zinc-800">
            <img src="/vital_logo.png" alt="VITAL Logo" className="w-full h-full object-cover" />
          </div>
          <div>
            <h1 className="text-xl font-extrabold tracking-tight text-white flex items-center gap-2.5">
              VITAL
            </h1>
            <p className="text-[10px] text-zinc-500 uppercase tracking-widest font-bold mt-0.5">Vision-Based Intelligent Triage &amp; Autonomous Lifesign Analytics</p>
          </div>
        </div>

        {/* Server status monitor */}
        <div className="flex items-center gap-4">
          {/* ARIA indicator */}
          <div className="hidden md:flex items-center gap-2 text-xs bg-[#0b1022] border border-indigo-800/40 rounded-lg px-3 py-1.5">
            <span className="w-2 h-2 rounded-full bg-indigo-400 animate-pulse"></span>
            <span className="text-indigo-400 font-black uppercase tracking-widest text-[10px]">ARIA ONLINE</span>
          </div>
          <div className="flex items-center gap-2 text-sm bg-[#0b1022] border border-zinc-800 rounded-lg px-4 py-2">
            <span className="text-zinc-500 uppercase tracking-wider font-mono text-[10px] font-bold">System:</span>
            {backendOnline ? (
              <span className="flex items-center gap-1.5 font-extrabold text-emerald-400 text-xs">
                <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse shadow-[0_0_8px_rgba(52,211,153,0.5)]"></span>
                ONLINE
              </span>
            ) : (
              <span className="flex items-center gap-1.5 font-extrabold text-rose-500 text-xs">
                <span className="w-2 h-2 rounded-full bg-rose-500"></span>
                OFFLINE
              </span>
            )}
          </div>
          <div className="w-px h-6 bg-zinc-800"></div>
          <div className="text-xs bg-[#0b1022] border border-zinc-800 rounded-lg px-4 py-2 font-mono uppercase tracking-wider">
            <span className="text-zinc-500">Queue</span>
            <span className="font-black text-cyan-400 ml-2 text-sm">{triageQueue.length}</span>
          </div>
          <div className="w-px h-6 bg-zinc-800"></div>
          {/* Theme Toggle Button */}
          <button
            onClick={() => setTheme(prev => prev === 'dark' ? 'light' : 'dark')}
            className="flex items-center justify-center p-2 bg-[#0b1022] border border-zinc-800 rounded-lg hover:bg-zinc-800/50 hover:border-zinc-700 transition-colors"
            title={`Switch to ${theme === 'dark' ? 'Light' : 'Dark'} Mode`}
          >
            {theme === 'dark' ? (
              <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" className="text-amber-400">
                <circle cx="12" cy="12" r="5"/>
                <line x1="12" y1="1" x2="12" y2="3"/>
                <line x1="12" y1="21" x2="12" y2="23"/>
                <line x1="4.22" y1="4.22" x2="5.64" y2="5.64"/>
                <line x1="18.36" y1="18.36" x2="19.78" y2="19.78"/>
                <line x1="1" y1="12" x2="3" y2="12"/>
                <line x1="21" y1="12" x2="23" y2="12"/>
                <line x1="4.22" y1="19.78" x2="5.64" y2="18.36"/>
                <line x1="18.36" y1="5.64" x2="19.78" y2="4.22"/>
              </svg>
            ) : (
              <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" className="text-indigo-400">
                <path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"/>
              </svg>
            )}
          </button>
        </div>
      </header>

      {/* 2. Top-Level Tab Navigator */}
      <nav className="max-w-[1700px] mx-auto px-6 mt-5">
        <div className="flex bg-[#090e1e]/80 border border-zinc-800/80 rounded-xl p-1 backdrop-blur-xl shadow-lg gap-1">
          <button
            onClick={() => setActiveTab('monitor')}
            className={`flex-1 py-2.5 px-4 rounded-lg flex items-center justify-center gap-2 transition-all text-xs font-black uppercase tracking-widest ${
              activeTab === 'monitor' 
                ? 'bg-cyan-500/10 text-cyan-400 border border-cyan-500/30 shadow-glow-cyan' 
                : 'text-zinc-500 hover:text-zinc-300 border border-transparent hover:bg-zinc-800/30'
            }`}
          >
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5"><polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/></svg>
            Vitals Monitor
          </button>
          <button
            onClick={() => setActiveTab('queue')}
            className={`flex-1 py-2.5 px-4 rounded-lg flex items-center justify-center gap-2 transition-all text-xs font-black uppercase tracking-widest ${
              activeTab === 'queue' 
                ? 'bg-cyan-500/10 text-cyan-400 border border-cyan-500/30 shadow-glow-cyan' 
                : 'text-zinc-500 hover:text-zinc-300 border border-transparent hover:bg-zinc-800/30'
            }`}
          >
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5"><path d="M9 11l3 3L22 4"/><path d="M21 12v7a2 2 0 01-2 2H5a2 2 0 01-2-2V5a2 2 0 012-2h11"/></svg>
            Triage Dispatch
            {triageQueue.length > 0 && (
              <span className="bg-cyan-500 text-[#020408] text-[9px] font-black px-1.5 py-0.5 rounded-full leading-none">{triageQueue.length}</span>
            )}
          </button>
          <button
            onClick={() => setActiveTab('crew')}
            className={`flex-1 py-2.5 px-4 rounded-lg flex items-center justify-center gap-2 transition-all text-xs font-black uppercase tracking-widest ${
              activeTab === 'crew' 
                ? 'bg-cyan-500/10 text-cyan-400 border border-cyan-500/30 shadow-glow-cyan' 
                : 'text-zinc-500 hover:text-zinc-300 border border-transparent hover:bg-zinc-800/30'
            }`}
          >
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5"><circle cx="12" cy="8" r="4"/><path d="M4 20c0-4 3.6-7 8-7s8 3 8 7"/></svg>
            Clinical Crew
          </button>
          <button
            onClick={() => setActiveTab('chat')}
            className={`flex-1 py-2.5 px-4 rounded-lg flex items-center justify-center gap-2 transition-all text-xs font-black uppercase tracking-widest ${
              activeTab === 'chat' 
                ? 'bg-indigo-500/10 text-indigo-400 border border-indigo-500/30 shadow-[0_0_15px_rgba(99,102,241,0.15)]' 
                : 'text-zinc-500 hover:text-zinc-300 border border-transparent hover:bg-zinc-800/30'
            }`}
          >
            <span className="w-2 h-2 rounded-full bg-indigo-400 animate-pulse"></span>
            ARIA Assistant
          </button>
        </div>
      </nav>

      {/* 3. Main Workspace Container */}
      <main className="max-w-[1700px] mx-auto px-6 mt-6 pb-10">
        
        {/* VIEW 1: LIVE VITAL MONITOR */}
        {activeTab === 'monitor' && (
          <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-stretch">
            
            {/* Left Hand side: Camera Feed Controls (expanded to col-span-7) */}
            <div className="lg:col-span-7 flex flex-col gap-6">
              <div className="bg-[#090e1e]/60 border border-zinc-800/80 rounded-2xl p-6 backdrop-blur-xl shadow-xl flex flex-col relative overflow-hidden">
                <h2 className="text-xs font-black text-zinc-400 uppercase tracking-widest mb-4 flex items-center gap-2.5">
                  <span className="w-2.5 h-2.5 rounded-full bg-cyan-400 animate-ping"></span>
                  Intake Acquisition Feed
                </h2>

                <div className="relative min-h-[420px] bg-[#02040a] border border-zinc-800 rounded-xl overflow-hidden mb-4 group flex items-center justify-center shadow-inner">
                  {backendOnline ? (
                    metrics.status === 'IMAGE_READY' || metrics.remark === 'IMAGE_DEMO' ? (
                      <img 
                        src={`${BACKEND_URL}/image_feed`} 
                        alt="Intake snap" 
                        className="w-full h-full object-contain"
                      />
                    ) : (
                      <img 
                        key={videoFeedKey}
                        src={`${BACKEND_URL}/video_feed?t=${videoFeedKey}`} 
                        alt="Medical Stream" 
                        className="w-full h-full object-contain"
                        onError={() => {
                          // Auto-retry connection after 1s if stream breaks
                          setTimeout(() => setVideoFeedKey(k => k + 1), 1000);
                        }}
                      />
                    )
                  ) : (
                    <div className="text-center p-6 text-zinc-500 max-w-[320px]">
                      <div className="text-3xl mb-3">📡</div>
                      <p className="text-sm font-bold text-zinc-300">Intake Hardware Offline</p>
                      <p className="text-xs text-zinc-500 mt-2 leading-relaxed">Ensure Python backend API server is running on port 5002 with required dependencies.</p>
                    </div>
                  )}

                  {/* Laser scan overlay */}
                  {(metrics.status === 'CALIBRATING' || metrics.status === 'OK') && (
                    <div className="absolute inset-0 pointer-events-none overflow-hidden">
                      <div className="w-full h-0.5 bg-gradient-to-r from-transparent via-cyan-400 to-transparent shadow-[0_0_10px_#06b6d4] opacity-60 absolute animate-scan"></div>
                    </div>
                  )}

                  {/* Floating acquisition state tags */}
                  <div className="absolute top-4 left-4 flex flex-wrap gap-2 pointer-events-none">
                    <span className={`text-[10px] font-extrabold uppercase px-2.5 py-1 rounded-md border ${
                      metrics.status === 'OK' ? 'bg-emerald-950/90 text-emerald-400 border-emerald-800/80 shadow-[0_0_10px_rgba(52,211,153,0.2)]' :
                      metrics.status === 'CALIBRATING' ? 'bg-cyan-950/90 text-cyan-400 border-cyan-800/80 animate-pulse' :
                      metrics.status === 'VIDEO_ENDED' ? 'bg-zinc-900/90 text-zinc-400 border-zinc-700/80' :
                      metrics.status === 'IMAGE_READY' ? 'bg-purple-950/90 text-purple-400 border-purple-800/80' :
                      'bg-zinc-950/90 text-zinc-500 border-zinc-800'
                    }`}>
                      {metrics.status}
                    </span>
                    
                    {metrics.face_detected && (
                      <span className="text-[10px] bg-cyan-950/95 text-cyan-400 border border-cyan-800/80 px-2.5 py-1 rounded-md font-extrabold tracking-wider">
                        TARGET DETECTED
                      </span>
                    )}
                  </div>
                </div>

                {/* Controller section */}
                <div className="flex flex-col gap-4">
                  <div className="flex flex-col gap-1.5">
                    <label className="text-[10px] text-zinc-450 font-black uppercase tracking-wider">Patient Identification (Full Name)</label>
                    <input 
                      type="text" 
                      placeholder="Enter patient full name (e.g. John Doe)..." 
                      value={patientName}
                      onChange={(e) => setPatientName(e.target.value)}
                      className="bg-[#02040a] border border-zinc-800 rounded-lg text-sm p-2.5 text-zinc-300 outline-none focus:border-cyan-500 transition-colors"
                    />
                  </div>

                  <div className="grid grid-cols-2 gap-4">
                    <div className="flex flex-col gap-1.5">
                      <label className="text-[10px] text-zinc-450 font-black uppercase tracking-wider">Optical Device</label>
                      <select 
                        value={selectedCamera !== null ? selectedCamera : ''}
                        onChange={(e) => setSelectedCamera(Number(e.target.value))}
                        className="bg-[#02040a] border border-zinc-800 rounded-lg text-sm p-2.5 text-zinc-300 outline-none focus:border-cyan-500 transition-colors cursor-pointer"
                      >
                        {cameras.map((cam) => (
                          <option key={cam.index} value={cam.index}>{cam.label}</option>
                        ))}
                        {cameras.length === 0 && <option value="">No hardware found</option>}
                      </select>
                    </div>

                    <div className="flex flex-col gap-1.5">
                      <label className="text-[10px] text-zinc-450 font-black uppercase tracking-wider">Interface controls</label>
                      <div className="flex gap-2">
                        <button 
                          onClick={handleStartWebcam}
                          className="flex-1 bg-cyan-500/10 hover:bg-cyan-500/20 text-cyan-400 border border-cyan-500/40 hover:border-cyan-500/60 font-bold text-xs py-2.5 px-3 rounded-lg transition-all shadow-glow-cyan"
                        >
                          Init Sensors
                        </button>
                        <button 
                          onClick={metrics.is_live ? handleReleaseCamera : handleStartWebcam}
                          title={metrics.is_live ? "Release Hardware (Stop)" : "Initialize Hardware (Play)"}
                          className={`px-4 py-2 border rounded-lg transition-colors flex items-center justify-center font-bold text-xs ${
                            metrics.is_live 
                              ? 'bg-rose-950/20 hover:bg-rose-900/20 border-rose-800/40 text-rose-400 shadow-[0_0_10px_rgba(244,63,94,0.1)]' 
                              : 'bg-emerald-950/20 hover:bg-emerald-900/20 border-emerald-800/40 text-emerald-400 shadow-[0_0_10px_rgba(52,211,153,0.1)]'
                          }`}
                        >
                          {metrics.is_live ? '⏹' : '▶'}
                        </button>
                      </div>
                    </div>
                  </div>

                  <div className="flex flex-col gap-1.5">
                    <label className="text-[10px] text-zinc-450 font-black uppercase tracking-wider">Upload Record (Video / Image)</label>
                    <div className="relative border border-dashed border-zinc-850 hover:border-zinc-700 bg-[#02040a]/40 hover:bg-[#02040a]/75 rounded-lg p-4 text-center cursor-pointer transition-all">
                      <input 
                        type="file" 
                        accept="video/*,image/*" 
                        onChange={handleFileUpload}
                        className="absolute inset-0 opacity-0 cursor-pointer w-full h-full"
                      />
                      <div className="text-zinc-300 text-sm font-semibold">
                        📂 Choose patient media file...
                      </div>
                      <p className="text-[9px] text-zinc-550 mt-1 uppercase tracking-wider font-mono">Supports MP4, webm, jpg, png</p>
                    </div>
                  </div>

                  {uploadMessage.text && (
                    <div className={`text-xs p-3 rounded-md font-bold ${
                      uploadMessage.type === 'success' ? 'bg-emerald-950/40 text-emerald-400 border border-emerald-900/30' : 'bg-rose-950/40 text-rose-400 border border-rose-900/30'
                    }`}>
                      {uploadMessage.text}
                    </div>
                  )}

                  {/* ── Signal Telemetry & Diagnostics ── unique hospital-grade widget */}
                  <div className="mt-2 bg-[#02040a]/60 border border-zinc-800/60 rounded-xl p-4">
                    <div className="text-[9px] text-zinc-500 font-black uppercase tracking-widest mb-3 flex items-center justify-between">
                      <div className="flex items-center gap-1.5">
                        <span className="w-1.5 h-1.5 rounded-full bg-cyan-400 animate-pulse"></span>
                        Signal Telemetry &amp; Diagnostics
                      </div>
                      <span className="font-mono text-[8px] text-zinc-650 bg-[#070b19] px-1.5 py-0.5 rounded border border-zinc-800">CHROM v2</span>
                    </div>

                    <div className="space-y-2 font-mono text-[10px]">
                      <div className="flex justify-between items-center py-1 border-b border-zinc-800/30">
                        <span className="text-zinc-500 uppercase">Camera Status</span>
                        <span className={`font-black uppercase ${metrics.is_live ? 'text-emerald-450' : 'text-zinc-600'}`}>
                          {metrics.is_live ? 'ACTIVE (30 FPS)' : 'OFFLINE'}
                        </span>
                      </div>
                      <div className="flex justify-between items-center py-1 border-b border-zinc-800/30">
                        <span className="text-zinc-500 uppercase">Face ROI Lock</span>
                        <span className={`font-black uppercase ${metrics.face_detected ? 'text-cyan-450' : 'text-zinc-600'}`}>
                          {metrics.face_detected ? 'LOCKED (142x44px)' : 'NO LOCK'}
                        </span>
                      </div>
                      <div className="flex justify-between items-center py-1 border-b border-zinc-800/30">
                        <span className="text-zinc-500 uppercase">Ambient Light</span>
                        <span className={`font-black uppercase ${metrics.estimated_lux > 100 ? 'text-emerald-450' : 'text-amber-450'}`}>
                          {metrics.estimated_lux} LUX ({metrics.estimated_lux > 100 ? 'OPTIMAL' : 'LOW'})
                        </span>
                      </div>
                      <div className="flex justify-between items-center py-1 border-b border-zinc-800/30">
                        <span className="text-zinc-500 uppercase">Signal Stability</span>
                        <span className="text-white font-black">{metrics.stability > 0 ? `${metrics.stability.toFixed(0)}%` : '--'}</span>
                      </div>
                      <div className="flex justify-between items-center py-1">
                        <span className="text-zinc-500 uppercase">Noise Level (SNR)</span>
                        <span className="text-indigo-400 font-black">{metrics.snr_db > 0 ? `${Number(metrics.snr_db).toFixed(1)} dB` : '--'}</span>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            </div>

            {/* Right Hand side: Vital Signs grid & PPG Plot (shrunk to col-span-5) */}
            <div className="lg:col-span-5 flex flex-col gap-6">
              
              {isTriageRunning && (
                <div className="bg-[#090e1e]/80 border border-indigo-500/40 rounded-2xl p-5 backdrop-blur-xl shadow-xl flex items-center gap-4 animate-pulse">
                  <div className="w-8 h-8 rounded-full border-2 border-indigo-400 border-t-transparent animate-spin shrink-0"></div>
                  <div>
                    <h3 className="text-xs font-black text-indigo-400 uppercase tracking-widest">Clinical Crew Running</h3>
                    <p className="text-[10px] text-zinc-450 uppercase font-bold mt-1 leading-relaxed">
                      3-Agent pipeline is executing diagnostics, ESI acuity check, and saving record to Neon.
                    </p>
                  </div>
                </div>
              )}
              
              {/* VITALS GRID */}
              <div className="bg-[#090e1e]/60 border border-zinc-800/80 rounded-2xl p-6 backdrop-blur-xl shadow-xl">
                <div className="flex items-center justify-between mb-4">
                  <h2 className="text-xs font-black text-zinc-400 uppercase tracking-widest flex items-center gap-2">
                    <span className="w-2.5 h-2.5 rounded-full bg-emerald-400"></span>
                    Optical Physiology Indicators
                  </h2>
                  {/* 30-sec session acquisition timer */}
                  <div className="flex items-center gap-2 ml-auto">
                    {metrics.calibration_done && sessionTimer > 0 && (
                      <div className="flex items-center gap-2 bg-[#02040a] border border-cyan-800/40 rounded-lg px-3 py-1.5">
                        <svg width="28" height="28" viewBox="0 0 36 36" className="-rotate-90">
                          <circle cx="18" cy="18" r="14" fill="none" stroke="rgba(6,182,212,0.15)" strokeWidth="3"/>
                          <circle cx="18" cy="18" r="14" fill="none" stroke="#06b6d4" strokeWidth="3"
                            strokeDasharray={`${2 * Math.PI * 14}`}
                            strokeDashoffset={`${2 * Math.PI * 14 * (1 - sessionTimer / 30)}`}
                            strokeLinecap="round" style={{transition: 'stroke-dashoffset 1s linear'}}/>
                          <text x="18" y="23" textAnchor="middle" fill="#06b6d4" fontSize="10" fontWeight="900"
                            style={{transform: 'rotate(90deg)', transformOrigin: '18px 18px'}}>{sessionTimer}</text>
                        </svg>
                        <div className="text-right">
                          <div className="text-[9px] text-zinc-500 font-black uppercase tracking-widest">Scan Window</div>
                          <div className="text-[11px] text-cyan-400 font-black">{sessionTimer}s remaining</div>
                        </div>
                      </div>
                    )}
                    {sessionTimer === 0 && !reportLoading && sessionReport && (
                      <div className="text-[10px] bg-emerald-950/60 border border-emerald-800/50 text-emerald-400 px-3 py-1.5 rounded-lg font-black uppercase tracking-widest flex items-center gap-1.5">
                        <span className="w-1.5 h-1.5 rounded-full bg-emerald-400"></span>
                        Report Ready
                      </div>
                    )}
                    {reportLoading && (
                      <div className="text-[10px] text-cyan-400 font-black uppercase tracking-widest flex items-center gap-1.5 animate-pulse">
                        <span className="w-1.5 h-1.5 rounded-full bg-cyan-400 animate-ping"></span>
                        Compiling Report...
                      </div>
                    )}
                  </div>
                </div>

                <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                  
                  {/* HEART RATE */}
                  <div className={`bg-[#02040a]/50 border rounded-xl p-5 flex flex-col justify-between hover:border-zinc-700 transition-all ${
                    metrics.classification === 'TACHYCARDIA' ? 'border-rose-500/40 bg-rose-950/5 shadow-glow-red' :
                    metrics.classification === 'NORMAL' ? 'border-emerald-500/20' :
                    'border-zinc-800'
                  }`}>
                    <div className="flex items-center justify-between">
                      <span className="text-[10px] text-zinc-400 font-extrabold uppercase tracking-wider">Heart Rate</span>
                      <span 
                        className="text-sm text-rose-500 transition-all transform origin-center animate-ping"
                        style={{ animationDuration: getHeartbeatDuration() }}
                      >
                        ❤️
                      </span>
                    </div>
                    <div className="my-3 flex items-baseline gap-2">
                      <span className="text-5xl font-black text-white tracking-tight leading-none font-mono">
                        {metrics.bpm > 0 ? metrics.bpm : '--'}
                      </span>
                      <span className="text-[11px] text-zinc-500 font-bold uppercase tracking-wider">BPM</span>
                    </div>
                    <div className="flex items-center justify-between border-t border-zinc-850 pt-2">
                      <span className={`text-[10px] px-2 py-0.5 rounded font-extrabold uppercase tracking-widest ${
                        metrics.classification === 'NORMAL' ? 'bg-emerald-950/80 text-emerald-450 border border-emerald-900/60' :
                        metrics.classification === 'TACHYCARDIA' ? 'bg-rose-950/80 text-rose-450 border border-rose-900/60 shadow-glow-red' :
                        metrics.classification === 'BRADYCARDIA' ? 'bg-cyan-950/80 text-cyan-450 border border-cyan-900/60' :
                        'bg-zinc-900 text-zinc-500'
                      }`}>
                        {metrics.classification}
                      </span>
                      <span className="text-[10px] text-zinc-500 font-mono">Conf: {typeof metrics.confidence === 'number' ? metrics.confidence.toFixed(1) : metrics.confidence}%</span>
                    </div>
                  </div>

                  {/* RESPIRATORY RATE */}
                  <div className={`bg-[#02040a]/50 border rounded-xl p-5 flex flex-col justify-between hover:border-zinc-700 transition-all ${
                    metrics.rr_classification === 'TACHYPNEA' ? 'border-rose-500/40 bg-rose-950/5' :
                    metrics.rr_classification === 'NORMAL' ? 'border-emerald-500/20' :
                    'border-zinc-800'
                  }`}>
                    <div className="flex items-center justify-between">
                      <span className="text-[10px] text-zinc-400 font-extrabold uppercase tracking-wider">Respiration</span>
                      <span className="text-sm text-cyan-450 animate-pulse">🌬️</span>
                    </div>
                    <div className="my-3 flex items-baseline gap-2 overflow-hidden">
                      <span className="text-5xl font-black text-white tracking-tight leading-none font-mono truncate">
                        {metrics.rr > 0 ? Number(metrics.rr).toFixed(1) : '--'}
                      </span>
                      <span className="text-[11px] text-zinc-500 font-bold uppercase tracking-wider shrink-0">B/min</span>
                    </div>
                    <div className="flex items-center justify-between border-t border-zinc-850 pt-2">
                      <span className={`text-[10px] px-2 py-0.5 rounded font-extrabold uppercase tracking-widest ${
                        metrics.rr_classification === 'NORMAL' ? 'bg-emerald-950/80 text-emerald-450 border border-emerald-900/60' :
                        metrics.rr_classification === 'TACHYPNEA' ? 'bg-rose-950/80 text-rose-450 border border-rose-900/60' :
                        metrics.rr_classification === 'BRADYPNEA' ? 'bg-cyan-950/80 text-cyan-450 border border-cyan-900/60' :
                        'bg-zinc-900 text-zinc-500'
                      }`}>
                        {metrics.rr_classification}
                      </span>
                      <span className="text-[10px] text-zinc-500 font-mono">Conf: {typeof metrics.rr_confidence === 'number' ? Number(metrics.rr_confidence).toFixed(1) : metrics.rr_confidence}%</span>
                    </div>
                  </div>

                  {/* STRESS INDEX */}
                  <div className={`bg-[#02040a]/50 border rounded-xl p-5 flex flex-col justify-between hover:border-zinc-700 transition-all ${
                    metrics.stress_index > 150 ? 'border-rose-500/40 bg-rose-950/5' :
                    metrics.stress_index > 80 ? 'border-amber-500/30 bg-amber-950/5' :
                    'border-zinc-800'
                  }`}>
                    <div className="flex items-center justify-between">
                      <span className="text-[10px] text-zinc-400 font-extrabold uppercase tracking-wider">Stress Index</span>
                      <span className="text-sm text-yellow-500">⚡</span>
                    </div>
                    <div className="my-3 flex items-baseline gap-2 overflow-hidden">
                      <span className={`font-black text-white tracking-tight leading-none font-mono ${
                        metrics.stress_index > 999 ? 'text-3xl' : 'text-5xl'
                      }`}>
                        {metrics.stress_index > 0 ? Number(metrics.stress_index).toFixed(0) : '--'}
                      </span>
                      <span className="text-[11px] text-zinc-500 font-bold uppercase tracking-wider shrink-0">INDEX</span>
                    </div>
                    <div className="flex items-center justify-between border-t border-zinc-850 pt-2">
                      <span className={`text-[10px] px-2 py-0.5 rounded font-extrabold uppercase tracking-widest ${
                        metrics.stress_index > 150 ? 'bg-rose-950/80 text-rose-450' :
                        metrics.stress_index > 80 ? 'bg-amber-950/80 text-amber-450' :
                        metrics.stress_index > 0 ? 'bg-emerald-950/80 text-emerald-450' :
                        'bg-zinc-900 text-zinc-500'
                      }`}>
                        {metrics.stress_index > 150 ? 'CRITICAL' : metrics.stress_index > 80 ? 'ELEVATED' : metrics.stress_index > 0 ? 'OPTIMAL' : '--'}
                      </span>
                      <span className="text-[10px] text-zinc-500 font-mono">HRV: {typeof metrics.hrv === 'number' ? Number(metrics.hrv).toFixed(0) : '--'}ms</span>
                    </div>
                  </div>

                </div>

                <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mt-4">
                  {/* SIGNAL QUALITY & SNR */}
                  <div className="bg-[#02040a]/50 border border-zinc-800 rounded-xl p-4 flex items-center justify-between hover:border-zinc-700 transition-all">
                    <div>
                      <span className="text-[10px] text-zinc-500 font-extrabold uppercase tracking-wider">Signal SNR</span>
                      <div className="text-2xl font-black text-white font-mono mt-1">{metrics.snr_db ? `${metrics.snr_db.toFixed(1)} dB` : '--'}</div>
                    </div>
                    <div className="text-right">
                      <span className="text-[10px] text-zinc-500 font-extrabold uppercase tracking-wider">Stability</span>
                      <div className="text-sm font-bold text-cyan-400 font-mono mt-1">{metrics.stability_indicator} ({metrics.stability.toFixed(1)} bpm)</div>
                    </div>
                  </div>

                  {/* LIGHT & MOTION */}
                  <div className="bg-[#02040a]/50 border border-zinc-800 rounded-xl p-4 flex items-center justify-between hover:border-zinc-700 transition-all">
                    <div>
                      <span className="text-[10px] text-zinc-500 font-extrabold uppercase tracking-wider">Acquisition Environment</span>
                      <div className="text-sm font-extrabold text-zinc-300 mt-1 flex flex-col gap-0.5">
                        <span>Luminance: <strong className="text-white font-mono">{metrics.estimated_lux} LUX</strong></span>
                        <span>Motion: <strong className="text-white font-mono">{metrics.motion_delta.toFixed(1)}</strong></span>
                      </div>
                    </div>
                    <div className="text-[10px] text-zinc-550 text-right leading-relaxed font-semibold">
                      <div>LIMIT: &gt;100 LUX</div>
                      <div>MOTION: &lt;15.0</div>
                    </div>
                  </div>
                </div>

                {metrics.warnings && metrics.warnings.length > 0 && (
                  <div className="mt-4 p-4 bg-rose-950/20 border border-rose-900/40 rounded-xl flex flex-col gap-1.5 shadow-glow-red animate-pulse">
                    <div className="text-xs font-black text-rose-400 uppercase tracking-wider flex items-center gap-1.5">
                      ⚠ Telemetry acquisition warnings:
                    </div>
                    <ul className="list-disc list-inside text-xs text-rose-300 font-semibold leading-relaxed">
                      {metrics.warnings.map((w, idx) => <li key={idx}>{w}</li>)}
                    </ul>
                  </div>
                )}
              </div>

              {/* 30-SECOND SESSION REPORT */}
              {sessionReport && (
                <div className="bg-[#090e1e]/80 border border-emerald-800/40 rounded-2xl p-5 backdrop-blur-xl shadow-xl animate-in fade-in duration-500">
                  <div className="flex items-center justify-between mb-4">
                    <div className="flex items-center gap-2">
                      <span className="w-2 h-2 rounded-full bg-emerald-400"></span>
                      <h3 className="text-xs font-black text-emerald-400 uppercase tracking-widest">30-Second Clinical Session Report</h3>
                    </div>
                    <div className="flex items-center gap-2">
                      <span className="text-[9px] text-zinc-500 font-mono uppercase">{sessionReport.generated_at}</span>
                      <button onClick={() => { setSessionReport(null); setSessionTimer(30); reportFetchedRef.current = false; }}
                        className="text-zinc-600 hover:text-zinc-400 text-xs px-2 py-0.5 rounded border border-zinc-800 hover:border-zinc-700 transition-colors font-mono">
                        ✕
                      </button>
                    </div>
                  </div>

                  <div className="grid grid-cols-2 md:grid-cols-4 gap-3 mb-4">
                    <div className="bg-[#02040a] border border-zinc-800 rounded-xl p-3 text-center">
                      <div className="text-[9px] text-zinc-500 uppercase tracking-widest font-bold mb-1">Avg BPM</div>
                      <div className="text-2xl font-black font-mono text-white">{sessionReport.vitals.heart_rate_avg ?? '--'}</div>
                      <div className={`text-[9px] mt-1 font-black uppercase tracking-widest ${
                        sessionReport.vitals.classification === 'NORMAL' ? 'text-emerald-400' :
                        sessionReport.vitals.classification === 'TACHYCARDIA' ? 'text-rose-400' : 'text-cyan-400'
                      }`}>{sessionReport.vitals.classification}</div>
                    </div>
                    <div className="bg-[#02040a] border border-zinc-800 rounded-xl p-3 text-center">
                      <div className="text-[9px] text-zinc-500 uppercase tracking-widest font-bold mb-1">Respiration</div>
                      <div className="text-2xl font-black font-mono text-white">{sessionReport.vitals.respiratory_rate ?? '--'}</div>
                      <div className="text-[9px] mt-1 font-black uppercase tracking-widest text-cyan-400">{sessionReport.vitals.rr_classification}</div>
                    </div>
                    <div className="bg-[#02040a] border border-zinc-800 rounded-xl p-3 text-center">
                      <div className="text-[9px] text-zinc-500 uppercase tracking-widest font-bold mb-1">HRV (RMSSD)</div>
                      <div className="text-2xl font-black font-mono text-indigo-400">{sessionReport.vitals.hrv_rmssd_ms ?? '--'}</div>
                      <div className="text-[9px] mt-1 font-bold text-zinc-500 uppercase">ms</div>
                    </div>
                    <div className="bg-[#02040a] border border-zinc-800 rounded-xl p-3 text-center">
                      <div className="text-[9px] text-zinc-500 uppercase tracking-widest font-bold mb-1">Stress</div>
                      <div className={`text-2xl font-black font-mono ${
                        sessionReport.vitals.stress_label === 'CRITICAL' ? 'text-rose-400' :
                        sessionReport.vitals.stress_label === 'ELEVATED' ? 'text-amber-400' : 'text-emerald-400'
                      }`}>{sessionReport.vitals.stress_index ?? '--'}</div>
                      <div className={`text-[9px] mt-1 font-black uppercase tracking-widest ${
                        sessionReport.vitals.stress_label === 'CRITICAL' ? 'text-rose-400' :
                        sessionReport.vitals.stress_label === 'ELEVATED' ? 'text-amber-400' : 'text-emerald-400'
                      }`}>{sessionReport.vitals.stress_label}</div>
                    </div>
                  </div>

                  <div className="bg-[#02040a]/80 border border-zinc-800/60 rounded-xl p-4 mb-3">
                    <div className="text-[9px] text-zinc-500 font-black uppercase tracking-widest mb-2">Clinical Interpretation</div>
                    <p className="text-sm text-zinc-300 leading-relaxed font-semibold">{sessionReport.clinical_summary}</p>
                  </div>

                  <div className="flex flex-wrap items-center gap-4 text-[9px] text-zinc-600 font-bold uppercase tracking-wider">
                    <span>Confidence: <strong className="text-zinc-400">{sessionReport.signal_quality.confidence_pct}%</strong></span>
                    <span>SNR: <strong className="text-zinc-400">{sessionReport.signal_quality.snr_db ?? '--'} dB</strong></span>
                    <span>Stability: <strong className="text-zinc-400">{sessionReport.signal_quality.stability}</strong></span>
                    <span>Lux: <strong className="text-zinc-400">{sessionReport.signal_quality.luminance_lux}</strong></span>
                    <span className="text-zinc-700">· {sessionReport.disclaimer}</span>
                  </div>
                </div>
              )}

              {/* RPPG CANVAS GRAPH */}

              <div className="bg-[#090e1e]/60 border border-zinc-800/80 rounded-2xl p-6 backdrop-blur-xl shadow-xl flex-1 flex flex-col relative overflow-hidden">
                <div className="flex items-center justify-between mb-3">
                  <h2 className="text-xs font-black text-zinc-400 uppercase tracking-widest flex items-center gap-2">
                    <span className="w-2.5 h-2.5 rounded-full bg-cyan-400 animate-pulse shadow-glow-cyan"></span>
                    rPPG Photonic Waveform analysis
                  </h2>
                  <span className="text-xs text-zinc-500 font-mono tracking-widest uppercase">
                    {metrics.sqi > 0 ? `SQI: ${metrics.sqi}%` : 'Calibrating signals...'}
                  </span>
                </div>
                
                <div className="relative bg-[#02040a] border border-zinc-800 rounded-xl p-2 flex items-center justify-center overflow-hidden flex-1 min-h-[220px]">
                  <canvas 
                    ref={canvasRef} 
                    width={640} 
                    height={220} 
                    className="w-full h-full block"
                  />
                  
                  {metrics.status === 'CALIBRATING' && (
                    <div className="absolute inset-0 bg-[#02040ad0]/95 flex flex-col items-center justify-center p-4">
                      <div className="w-full max-w-sm bg-zinc-900 rounded-full h-2 overflow-hidden border border-zinc-800 mb-4 relative">
                        <div 
                          className="bg-gradient-to-r from-cyan-400 to-indigo-500 h-full rounded-full transition-all duration-300 shadow-[0_0_10px_rgba(6,182,212,0.6)]"
                          style={{ width: `${metrics.calibration_progress}%` }}
                        ></div>
                      </div>
                      <span className="text-xs text-cyan-400 font-black tracking-widest animate-pulse uppercase">
                        Calibrating Photonic Sensors ({metrics.calibration_progress}%)
                      </span>
                    </div>
                  )}
                </div>
                
                <div className="flex items-center justify-between text-[10px] text-zinc-550 mt-2 font-mono uppercase tracking-widest font-bold">
                  <span>0.00s</span>
                  <span>10s Rolling Sensor Buffer</span>
                  <span>10.00s</span>
                </div>
              </div>

            </div>

          </div>
        )}

        {/* VIEW 2: CENTRAL TRIAGE QUEUE */}
        {activeTab === 'queue' && (
          <div className="bg-[#090e1e]/60 border border-zinc-800/80 rounded-2xl p-6 backdrop-blur-xl shadow-xl flex flex-col min-h-[550px] relative overflow-hidden">
            
            <div className="flex items-center justify-between border-b border-zinc-800 pb-4 mb-6">
              <div>
                <h2 className="text-base font-black text-white uppercase tracking-wider flex items-center gap-2">
                  <span className="w-2.5 h-2.5 rounded-full bg-cyan-400 shadow-glow-cyan"></span>
                  Central Dispatch Triage Queue
                </h2>
                <p className="text-xs text-zinc-450 uppercase tracking-wider font-bold mt-1">Dynamically sorted by Acuity (ESI 1 & 2 prioritize to the top)</p>
              </div>
              <button 
                onClick={handleClearQueue}
                className="text-rose-450 hover:text-rose-400 hover:bg-rose-950/40 border border-rose-900/40 text-xs font-black uppercase tracking-widest py-2 px-5 rounded-lg transition-colors"
              >
                Flush Queue
              </button>
            </div>

            <div className="flex-1 overflow-y-auto flex flex-col gap-4 pr-1">
              {triageQueue.length === 0 ? (
                <div className="h-full flex flex-col items-center justify-center text-zinc-500 py-24 text-center text-sm font-bold uppercase tracking-widest">
                  <div className="text-4xl mb-3">📋</div>
                  <p className="text-zinc-300">Clinical Queue Empty</p>
                  <p className="text-[10px] text-zinc-550 mt-1 uppercase font-mono">Processed ESI patient records will compile here.</p>
                </div>
              ) : (
                triageQueue.map((patient) => (
                  <div 
                    key={patient.id} 
                    className={`border border-zinc-800 rounded-xl p-5 bg-[#02040a]/40 hover:bg-[#02040a]/75 transition-all flex flex-col md:flex-row md:items-center justify-between gap-6 relative overflow-hidden group ${
                      patient.is_shock ? 'border-rose-500/50 shadow-glow-red bg-rose-950/5' : 'hover:border-zinc-700'
                    }`}
                  >
                    {/* Urgency Sidebar Indicator */}
                    <div className={`absolute left-0 top-0 bottom-0 w-1.5 ${
                      patient.esi_level === 1 ? 'bg-red-500' :
                      patient.esi_level === 2 ? 'bg-orange-500' :
                      patient.esi_level === 3 ? 'bg-yellow-500' :
                      patient.esi_level === 4 ? 'bg-emerald-500' :
                      'bg-cyan-500'
                    }`}></div>

                    <div className="flex-1 pl-3">
                      <div className="flex flex-wrap items-center gap-4">
                        <span className="font-black text-white text-base">{patient.name}</span>
                        <span className="text-xs font-mono text-zinc-550 font-bold uppercase">{patient.timestamp}</span>
                        {patient.is_shock && (
                          <span className="text-[10px] bg-red-950 text-red-400 border border-red-900/60 px-2.5 py-0.5 rounded font-black uppercase tracking-widest animate-pulse shadow-glow-red">
                            ⚠️ COMPENSATED SHOCK ALERT
                          </span>
                        )}
                      </div>

                      <div className="mt-2 text-sm text-zinc-300 leading-relaxed max-w-[1200px] font-sans font-medium">
                        {patient.triage_summary}
                      </div>

                      <div className="mt-3.5 flex flex-wrap items-center gap-6 text-[10px] text-zinc-550 font-bold uppercase tracking-wider">
                        <span>Record ID: <strong className="text-zinc-300 font-mono">{patient.id}</strong></span>
                        <span>Acquisition Source: <strong className="text-zinc-300 font-mono">{patient.video_path}</strong></span>
                        <span>Diagnosis Target: <strong className="text-indigo-400">{patient.primary_diagnosis}</strong></span>
                      </div>
                    </div>

                    <div className="flex flex-row md:flex-col items-center gap-3 self-start md:self-auto">
                      <div className={`border rounded-lg py-2 px-4 text-center min-w-[95px] ${getEsiClass(patient.esi_level)}`}>
                        <div className="text-[9px] uppercase font-black tracking-widest opacity-85">ESI Level</div>
                        <div className="text-2xl font-black font-mono leading-none mt-1">{patient.esi_level}</div>
                      </div>
                      <div className="bg-[#090e1e] border border-zinc-800 text-zinc-300 font-mono text-[10px] px-3 py-1.5 rounded text-center min-w-[95px] font-bold uppercase tracking-wider">
                        Score: {patient.priority_score}
                      </div>
                    </div>
                  </div>
                ))
              )}
            </div>

          </div>
        )}

        {/* VIEW 3: AGENT CREW PANEL */}
        {activeTab === 'crew' && (
          <div className="bg-[#090e1e]/60 border border-zinc-800/80 rounded-2xl p-6 backdrop-blur-xl shadow-xl flex flex-col min-h-[550px] relative overflow-hidden">
            
            <div className="flex items-center justify-between border-b border-zinc-800 pb-4 mb-6">
              <div>
                <h2 className="text-base font-black text-white uppercase tracking-wider flex items-center gap-2">
                  <span className="w-2.5 h-2.5 rounded-full bg-purple-500 shadow-[0_0_8px_rgba(139,92,246,0.6)] animate-pulse"></span>
                  Multi-Agent Clinical Intelligence Crew
                </h2>
                <p className="text-xs text-zinc-450 uppercase tracking-wider font-bold mt-1">Execution logs of the 3-Agent Decoupled Triage pipeline</p>
              </div>

              <button 
                onClick={() => handleRunTriage()}
                disabled={isTriageRunning || metrics.status === 'DISCONNECTED'}
                className="bg-gradient-to-r from-cyan-500 to-indigo-600 hover:from-cyan-600 hover:to-indigo-700 disabled:from-zinc-900 disabled:to-zinc-900 disabled:text-zinc-650 disabled:cursor-not-allowed text-white font-extrabold text-sm py-3 px-6 rounded-xl transition-all shadow-glow-cyan hover:shadow-[0_0_20px_rgba(6,182,212,0.3)] flex items-center gap-2.5 uppercase tracking-wider"
              >
                {isTriageRunning ? (
                  <>
                    <span className="w-4 h-4 rounded-full border-2 border-white border-t-transparent animate-spin"></span>
                    Negotiating...
                  </>
                ) : (
                  <>
                    ⚡ Kickoff Triage Crew
                  </>
                )}
              </button>
            </div>

            {/* Agent selecting buttons */}
            <div className="flex border-b border-zinc-800 mb-5 bg-[#030612]/30 rounded-t-xl overflow-hidden">
              <button 
                onClick={() => setActiveAgentTab('perception')}
                className={`flex-1 py-3.5 text-xs font-black uppercase tracking-widest transition-all ${
                  activeAgentTab === 'perception' ? 'text-cyan-400 border-b-2 border-cyan-500 bg-[#0e1326]/40 font-black' : 'text-zinc-500 hover:text-zinc-300'
                }`}
              >
                Perception Agent (Vitals Capture)
              </button>
              <button 
                onClick={() => setActiveAgentTab('diagnostic')}
                className={`flex-1 py-3.5 text-xs font-black uppercase tracking-widest transition-all ${
                  activeAgentTab === 'diagnostic' ? 'text-amber-400 border-b-2 border-amber-500 bg-[#0e1326]/40 font-black' : 'text-zinc-500 hover:text-zinc-300'
                }`}
              >
                Diagnostic Agent (Acuity Assessment)
              </button>
              <button 
                onClick={() => setActiveAgentTab('coordinator')}
                className={`flex-1 py-3.5 text-xs font-black uppercase tracking-widest transition-all ${
                  activeAgentTab === 'coordinator' ? 'text-indigo-400 border-b-2 border-indigo-500 bg-[#0e1326]/40 font-black' : 'text-zinc-500 hover:text-zinc-300'
                }`}
              >
                Coordinator Agent (Queue Placement)
              </button>
            </div>

            {/* Main console screen */}
            <div className="bg-[#02040af0] border border-zinc-850 rounded-xl p-5 flex-1 overflow-y-auto max-h-[420px] font-mono text-sm leading-relaxed text-zinc-300 min-h-[300px] shadow-inner">
              {isTriageRunning && (
                <div className="h-full flex flex-col items-center justify-center text-zinc-500 py-16 gap-4">
                  <div className="w-10 h-10 rounded-full border-2 border-cyan-500 border-t-transparent animate-spin shadow-glow-cyan"></div>
                  <div className="text-center font-sans">
                    <p className="text-sm font-black text-zinc-200 uppercase tracking-wider">Multi-Agent Negotiation Active</p>
                    <p className="text-xs text-zinc-500 mt-2 max-w-[340px] leading-relaxed uppercase tracking-widest font-bold">Consolidating rPPG bio-telemetry, analyzing shock state indicators, and prioritizing patients...</p>
                  </div>
                </div>
              )}

              {!isTriageRunning && !lastTriageResult && (
                <div className="h-full flex items-center justify-center text-zinc-500 text-center py-20 font-sans uppercase tracking-widest text-xs font-bold">
                  <div>
                    <p>Sensor pipeline idle.</p>
                    <p className="text-zinc-650 mt-1.5 text-[10px] uppercase font-mono">Upload a record or start live webcam and trigger Triage Crew.</p>
                  </div>
                </div>
              )}

              {!isTriageRunning && lastTriageResult && (
                <div>
                  {activeAgentTab === 'perception' && (
                    <div className="flex flex-col gap-3">
                      <div className="text-zinc-400 border-b border-zinc-850 pb-3 mb-2 font-sans font-bold text-xs uppercase text-cyan-400 tracking-wider flex items-center justify-between">
                        <span>Perception Analysis Log</span>
                        <span>[COMPILED]</span>
                      </div>
                      <div className="whitespace-pre-wrap font-mono leading-relaxed">
                        {`Patient Record: ${lastTriageResult.name}\nTimestamp: ${lastTriageResult.timestamp}\n\nACQUIRED PHYSIOLOGY:\n- Path: ${lastTriageResult.video_path}\n- Core Metrics Resolved:\n  • Heart Rate: ${metrics.bpm} BPM\n  • Respiration Rate: ${metrics.rr} breaths/min\n  • HRV: ${metrics.hrv.toFixed(1)} ms\n  • Stress Index: ${metrics.stress_index.toFixed(0)}\n  • Signal SNR: ${metrics.snr_db.toFixed(1)} dB\n\nDiagnostic buffer synchronized.`}
                      </div>
                    </div>
                  )}

                  {activeAgentTab === 'diagnostic' && (
                    <div className="flex flex-col gap-3">
                      <div className="text-zinc-400 border-b border-zinc-850 pb-3 mb-2 font-sans font-bold text-xs uppercase text-amber-400 tracking-wider flex items-center justify-between">
                        <span>Clinical Diagnostic Assessment</span>
                        <span>[COMPILED]</span>
                      </div>
                      <div className="whitespace-pre-wrap font-mono leading-relaxed">
                        {`DIAGNOSIS PATHOLOGY:\n- Recommended Index: ESI LEVEL ${lastTriageResult.esi_level}\n- Clinical Focus: ${lastTriageResult.primary_diagnosis}\n- Compensated Shock: ${lastTriageResult.is_shock ? "⚠️ SHOCK CRITERIA SATISFIED" : "STABLE / NO SHOCK"}\n\nESI CORRELATION LOGIC:\n- Cross-correlation analysis: HR (${metrics.bpm}) × RR (${metrics.rr})\n- Clinical documentation resolved.\n\nRaw decision trace:\n${lastTriageResult.agent_output.substring(0, 1500)}...`}
                      </div>
                    </div>
                  )}

                  {activeAgentTab === 'coordinator' && (
                    <div className="flex flex-col gap-4 font-sans p-2">
                      <div className="text-zinc-400 border-b border-zinc-850 pb-3 mb-2 font-sans font-bold text-xs uppercase text-indigo-400 tracking-wider flex items-center justify-between">
                        <span>Dynamic Queue Allocation</span>
                        <span>[COMPILED]</span>
                      </div>
                      
                      <div className="grid grid-cols-2 gap-4">
                        <div className="bg-[#060813] border border-zinc-800 rounded-lg p-4 text-center shadow-glow-cyan">
                          <div className="text-[10px] text-zinc-500 font-extrabold uppercase tracking-wider">Acuity Level</div>
                          <div className="text-4xl font-black text-indigo-400 mt-2 font-mono">Level {lastTriageResult.esi_level}</div>
                        </div>
                        <div className="bg-[#060813] border border-zinc-800 rounded-lg p-4 text-center">
                          <div className="text-[10px] text-zinc-500 font-extrabold uppercase tracking-wider">Priority Rating</div>
                          <div className="text-4xl font-black text-cyan-400 mt-2 font-mono">{lastTriageResult.priority_score}/100</div>
                        </div>
                      </div>

                      <div className="bg-[#060813] border border-zinc-800 rounded-lg p-4">
                        <div className="text-[10px] text-zinc-500 font-extrabold uppercase tracking-wider mb-2 font-bold">Primary Clinical Indicator</div>
                        <div className="flex items-center gap-3">
                          <span className={`w-3.5 h-3.5 rounded-full ${lastTriageResult.is_shock ? 'bg-red-500 animate-ping shadow-[0_0_8px_#ef4444]' : 'bg-emerald-500'}`}></span>
                          <span className={`text-sm font-bold ${lastTriageResult.is_shock ? 'text-rose-400 font-black' : 'text-zinc-200'}`}>
                            {lastTriageResult.primary_diagnosis} {lastTriageResult.is_shock && "(Shock Indicator Active)"}
                          </span>
                        </div>
                      </div>

                      <div className="bg-[#02040a] border border-zinc-800 rounded-lg p-4">
                        <div className="text-[10px] text-zinc-500 font-extrabold uppercase tracking-wider mb-2 font-bold">Triage Summary & Rationale</div>
                        <p className="text-sm leading-relaxed text-zinc-200 font-medium">{lastTriageResult.triage_summary}</p>
                      </div>
                    </div>
                  )}
                </div>
              )}
            </div>

          </div>
        )}

        {/* VIEW 4: ARIA CLINICAL ASSISTANT */}
        {activeTab === 'chat' && (
          <div className="bg-[#090e1e]/60 border border-zinc-800/80 rounded-2xl p-6 backdrop-blur-xl shadow-xl flex flex-col min-h-[600px] relative overflow-hidden">
            
            {/* ARIA Header */}
            <div className="flex items-center justify-between border-b border-zinc-800/80 pb-5 mb-5">
              <div className="flex items-center gap-4">
                <div className="w-12 h-12 rounded-xl bg-gradient-to-tr from-indigo-500 via-purple-500 to-pink-500 flex items-center justify-center shadow-lg shadow-indigo-500/20 relative overflow-hidden shrink-0">
                  <div className="absolute inset-0 bg-white/10 mix-blend-overlay"></div>
                  <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="white" strokeWidth="2"><path d="M12 2a10 10 0 110 20A10 10 0 0112 2z" opacity="0.3"/><path d="M12 6v6l4 2"/><circle cx="12" cy="12" r="2" fill="white"/></svg>
                </div>
                <div>
                  <div className="flex items-center gap-2">
                    <h2 className="text-lg font-black text-white tracking-tight">ARIA</h2>
                    <span className="text-[9px] font-extrabold uppercase tracking-widest bg-indigo-950/80 text-indigo-400 border border-indigo-800/60 px-2 py-0.5 rounded">Adaptive Real-time Intelligence for Acute-care</span>
                  </div>
                  <p className="text-[10px] text-zinc-500 font-bold uppercase tracking-widest mt-0.5">Powered by NVIDIA NIM · OCR Document Parsing · Multilingual · Evidence-Based</p>
                </div>
              </div>
              <div className="flex items-center gap-2 text-[10px] font-black uppercase tracking-widest">
                <span className="w-2 h-2 rounded-full bg-indigo-400 animate-pulse"></span>
                <span className="text-indigo-400">ARIA Online</span>
              </div>
            </div>

            {/* Chat message bubbles */}
            <div className="flex-1 bg-[#02040af0] border border-zinc-850 rounded-xl p-5 overflow-y-auto max-h-[360px] flex flex-col gap-4 mb-4 shadow-inner min-h-[300px]">
              {chatHistory.length === 0 ? (
                <div className="h-full flex flex-col items-center justify-center text-zinc-550 text-center py-16">
                  <div className="w-16 h-16 rounded-2xl bg-gradient-to-tr from-indigo-500/20 to-purple-500/10 border border-indigo-800/30 flex items-center justify-center mb-4 mx-auto">
                    <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="rgb(129,140,248)" strokeWidth="1.5"><path d="M21 15a2 2 0 01-2 2H7l-4 4V5a2 2 0 012-2h14a2 2 0 012 2z"/></svg>
                  </div>
                  <p className="text-zinc-200 font-black text-sm uppercase tracking-wider">ARIA Clinical Intelligence Ready</p>
                  <p className="text-[11px] text-zinc-500 mt-2 max-w-[460px] leading-relaxed font-semibold">Ask clinical questions, request differential diagnoses, interpret vital signs, or attach photos of patient records and lab reports for instant OCR analysis.</p>
                  <div className="mt-5 flex flex-wrap gap-2 justify-center max-w-[500px]">
                    {['What does BRADYPNEA indicate?', 'Explain ESI Level 2', 'Interpret high stress index', 'Normal rPPG heart rate range'].map(s => (
                      <button key={s} onClick={() => setChatInput(s)} className="text-[10px] bg-indigo-950/60 hover:bg-indigo-950/80 border border-indigo-800/40 text-indigo-300 px-3 py-1.5 rounded-lg font-bold uppercase tracking-wider transition-colors">{s}</button>
                    ))}
                  </div>
                </div>
              ) : (
                chatHistory.map((msg, idx) => (
                  <div key={idx} className={`flex flex-col max-w-[85%] ${msg.role === 'user' ? 'self-end items-end' : 'self-start items-start'}`}>
                    <span className="text-[9px] text-zinc-500 font-extrabold uppercase tracking-widest mb-1.5">
                      {msg.role === 'user' ? 'Clinician' : 'ARIA'}
                    </span>
                    
                    <div className={`p-4 rounded-xl text-sm leading-relaxed font-semibold whitespace-pre-wrap ${
                      msg.role === 'user' 
                        ? 'bg-indigo-950/80 text-zinc-100 border border-indigo-900/60 rounded-tr-none shadow-glow-cyan' 
                        : 'bg-zinc-900/60 text-zinc-250 border border-zinc-800 rounded-tl-none'
                    }`}>
                      {/* Attached Image inside Bubble */}
                      {msg.image && (
                        <div className="mb-3 max-w-[200px] rounded-lg overflow-hidden border border-zinc-850">
                          <img src={msg.image} alt="Clinical document" className="w-full h-auto object-cover" />
                        </div>
                      )}
                      {msg.content.replace(/^\[[a-z]{2}-[A-Z]{2}\]\s*/i, '').replace(/\*\*/g, '').replace(/\* /g, '• ')}
                    </div>
                  </div>
                ))
              )}
              
              {isChatLoading && (
                <div className="self-start flex items-center gap-2.5 text-xs text-cyan-400 uppercase tracking-widest font-black font-mono">
                  <span className="w-2.5 h-2.5 rounded-full bg-cyan-400 animate-ping"></span>
                  Decoding Telemetry...
                </div>
              )}
              <div ref={chatEndRef}></div>
            </div>

            {/* Input form */}
            <form onSubmit={handleSendChatMessage} className="flex flex-col gap-3">
              {/* Selected Image preview box */}
              {selectedImage && (
                <div className="flex items-center gap-3 p-3 bg-[#02040a]/60 border border-zinc-850 rounded-xl animate-pulse">
                  <div className="w-12 h-12 rounded-lg overflow-hidden border border-zinc-800 relative">
                    <img src={selectedImage} alt="Attachment Preview" className="w-full h-full object-cover" />
                  </div>
                  <div className="flex-1 min-w-0">
                    <p className="text-xs font-bold text-zinc-300 truncate">Clinical Document Attachment Loaded</p>
                    <p className="text-[9px] text-zinc-500 uppercase font-mono tracking-wider font-semibold">Ready for NVIDIA NIM OCR parsing</p>
                  </div>
                  <button 
                    type="button" 
                    onClick={() => setSelectedImage(null)}
                    className="text-rose-400 hover:text-rose-350 font-black text-sm p-2"
                  >
                    ✕ remove
                  </button>
                </div>
              )}

              <div className="flex gap-3">
                <input 
                  type="file" 
                  accept="image/*" 
                  ref={imageInputRef}
                  onChange={handleImageAttachment}
                  className="hidden"
                />
                <button 
                  type="button"
                  onClick={() => imageInputRef.current?.click()}
                  className="p-3.5 rounded-xl border border-zinc-800 bg-zinc-900 text-zinc-400 hover:text-zinc-200 hover:bg-zinc-850 transition-colors text-sm"
                  title="Attach Document Photo"
                >
                  📎 Attach
                </button>
                <button 
                  type="button"
                  onClick={toggleRecording}
                  className={`p-3.5 rounded-xl border transition-colors text-sm font-bold ${
                    isRecording ? 'bg-rose-950 text-rose-400 border-rose-800 animate-pulse' : 'bg-zinc-900 text-zinc-400 border-zinc-800 hover:bg-zinc-850'
                  }`}
                  title="Voice Input (STT)"
                >
                  🎙️ Speak
                </button>
                <input 
                  type="text"
                  value={chatInput}
                  onChange={(e) => setChatInput(e.target.value)}
                  placeholder={selectedImage ? "Add clinical context or ask ARIA to parse this document..." : "Ask ARIA about vital signs, diagnoses, ESI levels, or clinical protocols..."}
                  className="flex-1 bg-[#02040a] border border-zinc-800 rounded-xl text-sm px-4 text-zinc-150 outline-none focus:border-cyan-500 transition-colors"
                />
                <button 
                  type="submit"
                  disabled={isChatLoading || (!chatInput.trim() && !selectedImage)}
                  className="bg-indigo-950 hover:bg-indigo-900 text-indigo-400 border border-indigo-850 font-bold text-xs py-3 px-6 rounded-xl transition-colors disabled:opacity-50 disabled:cursor-not-allowed uppercase tracking-wider"
                >
                  Send
                </button>
              </div>
            </form>
          </div>
        )}

      </main>

    </div>
  );
}
