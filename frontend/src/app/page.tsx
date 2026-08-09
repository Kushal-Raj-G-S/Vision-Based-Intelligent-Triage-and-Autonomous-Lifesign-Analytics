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
  heart_rate?: number;
  respiration?: number;
  hrv?: number;
  stress_index?: number;
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
  video_current_sec?: number | null;
  video_total_sec?: number | null;
}

const BACKEND_URL = "http://127.0.0.1:5002";

type NavigationTab = 'monitor' | 'queue' | 'crew' | 'chat';

export default function Home() {
  // Navigation
  const [activeTab, setActiveTab] = useState<NavigationTab>('monitor');
  const [sidebarCollapsed, setSidebarCollapsed] = useState<boolean>(false);

  // Switch sections and always land at the top of the page, rather than
  // keeping whatever scroll position the previous section was left at.
  const goToTab = (tab: NavigationTab) => {
    setActiveTab(tab);
    window.scrollTo({ top: 0, behavior: 'auto' });
  };

  // "92.4" -> "1:32" -- used for the uploaded-video playback-progress overlay.
  const formatDuration = (seconds: number): string => {
    const total = Math.max(0, Math.round(seconds));
    const m = Math.floor(total / 60);
    const s = total % 60;
    return `${m}:${s.toString().padStart(2, '0')}`;
  };

  // Connection and settings state
  const [backendOnline, setBackendOnline] = useState<boolean>(false);
  const [cameras, setCameras] = useState<CameraDevice[]>([]);
  const [selectedCamera, setSelectedCamera] = useState<number | null>(null);
  // Optical Device mode: a live camera, or a previously uploaded video file
  // chosen explicitly from the list below rather than auto-playing on upload.
  const [deviceMode, setDeviceMode] = useState<'camera' | 'upload'>('camera');
  const [uploadedVideos, setUploadedVideos] = useState<{ filename: string; size_bytes: number; modified: string }[]>([]);
  const [selectedUploadedVideo, setSelectedUploadedVideo] = useState<string>('');
  
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
  const [expandedPatientId, setExpandedPatientId] = useState<string | null>(null);
  const [isTriageRunning, setIsTriageRunning] = useState<boolean>(false);
  const [lastTriageResult, setLastTriageResult] = useState<PatientRecord | null>(null);
  const [triageAutoRan, setTriageAutoRan] = useState<boolean>(false);
  const [selectedPatientId, setSelectedPatientId] = useState<string>('');
  const [activeAgentTab, setActiveAgentTab] = useState<'perception' | 'diagnostic' | 'coordinator'>('perception');
  
  // Chat state
  const [chatHistory, setChatHistory] = useState<ChatMessage[]>([]);
  const [chatInput, setChatInput] = useState<string>('');
  const [selectedImage, setSelectedImage] = useState<string | null>(null);
  const [isChatLoading, setIsChatLoading] = useState<boolean>(false);
  const [isRecording, setIsRecording] = useState<boolean>(false);
  // TTS is muted by default — ARIA never speaks a response unless the user
  // explicitly turns the volume toggle on.
  const [ttsEnabled, setTtsEnabled] = useState<boolean>(false);
  const voicesRef = useRef<SpeechSynthesisVoice[]>([]);

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
  const videoTriageFetchedRef = useRef<boolean>(false);
  // Synchronous lock against handleRunTriage firing twice in close succession
  // (e.g. the 30s auto-trigger and a manual click landing in the same tick).
  // isTriageRunning is React state and doesn't update synchronously, so it
  // can't prevent a second call that starts before the first render commits —
  // a ref can, since it's mutated immediately, not on next render.
  const triageInFlightRef = useRef<boolean>(false);
  
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
        fetchUploadedVideos();
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

  const fetchUploadedVideos = async () => {
    try {
      const res = await fetch(`${BACKEND_URL}/api/uploaded_videos`);
      if (res.ok) {
        const data = await res.json();
        const videos = data.videos || [];
        setUploadedVideos(videos);
        // Keep the current selection if it still exists; otherwise default
        // to the most recent upload so the dropdown isn't left empty.
        setSelectedUploadedVideo((prev) =>
          videos.some((v: { filename: string }) => v.filename === prev) ? prev : (videos[0]?.filename || '')
        );
      }
    } catch (e) {
      console.error("Error fetching uploaded videos:", e);
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

  // Scroll to bottom of chat — only when the chat itself changes, not on
  // every tab switch (that used to fight with goToTab's scroll-to-top).
  useEffect(() => {
    if (activeTab !== 'chat') return;
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
  }, [chatHistory, isChatLoading]);

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
                // Disable the manual "Kickoff" button THE MOMENT the countdown
                // ends, not after the report fetch resolves. It used to flip
                // only inside the .then() below, which left a multi-second
                // window (report fetch latency) where the button was still
                // enabled — a manual click in that window created a second,
                // duplicate queue entry for the same session under a
                // different (auto-generated) patient ID.
                setTriageAutoRan(true);
                setReportLoading(true);
                fetch(`${BACKEND_URL}/api/generate_report`, { method: 'POST' })
                  .then(r => r.json())
                  .then(d => {
                    if (d.success) {
                      setSessionReport(d.report);
                      // Auto-run Clinical Crew triage with custom patient name.
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
        setTriageAutoRan(false);
      }
    }
    return () => {};
  }, [metrics.calibration_done, metrics.is_live, sessionReport]);

  // Auto-trigger triage for UPLOADED VIDEOS once the whole file has been
  // processed (VIDEO_ENDED) -- the live-session countdown above only fires
  // for is_live sessions, on a fixed 30s window; a video has no fixed
  // window, but its natural end IS the "the whole session's data is ready"
  // signal, so trigger on that instead of leaving it purely manual.
  useEffect(() => {
    if (metrics.status === 'VIDEO_ENDED' && !metrics.is_live) {
      if (!videoTriageFetchedRef.current) {
        videoTriageFetchedRef.current = true;
        // Same synchronous-flip reasoning as the live countdown above --
        // disable the manual Kickoff button immediately, not after the
        // report fetch resolves, so a manual click can't race this and
        // create a duplicate queue entry for the same video.
        setTriageAutoRan(true);
        setReportLoading(true);
        fetch(`${BACKEND_URL}/api/generate_report`, { method: 'POST' })
          .then(r => r.json())
          .then(d => {
            if (d.success) {
              setSessionReport(d.report);
              handleRunTriage(patientName);
            }
          })
          .catch(console.error)
          .finally(() => setReportLoading(false));
      }
    } else if (metrics.status !== 'VIDEO_ENDED') {
      // Reset once a new session starts (a fresh video played, or a live
      // session begins) so the next VIDEO_ENDED can auto-trigger again.
      videoTriageFetchedRef.current = false;
    }
  }, [metrics.status, metrics.is_live]);

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

  // 3b. Load available TTS voices (populated asynchronously by the browser)
  useEffect(() => {
    if (typeof window === 'undefined' || !window.speechSynthesis) return;
    const loadVoices = () => { voicesRef.current = window.speechSynthesis.getVoices(); };
    loadVoices();
    window.speechSynthesis.onvoiceschanged = loadVoices;
    return () => { window.speechSynthesis.onvoiceschanged = null; };
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

  // 4. Voice TTS output — only ever called when the user has explicitly
  // turned the volume toggle on (see ttsEnabled), never automatically.
  const pickFemaleVoice = (): SpeechSynthesisVoice | null => {
    const voices = voicesRef.current;
    if (!voices.length) return null;
    const englishVoices = voices.filter(v => v.lang.toLowerCase().startsWith('en'));
    const pool = englishVoices.length ? englishVoices : voices;
    // Known good female voice names across Windows/Chrome/Edge/macOS, checked
    // in priority order — most natural-sounding first.
    const preferredNames = [
      'Google UK English Female', 'Google US English', 'Microsoft Aria',
      'Microsoft Jenny', 'Microsoft Zira', 'Samantha', 'Victoria', 'Female',
    ];
    for (const name of preferredNames) {
      const match = pool.find(v => v.name.toLowerCase().includes(name.toLowerCase()));
      if (match) return match;
    }
    // Fallback heuristic: many engines label gender in the voice name only
    // loosely — exclude names that are clearly male, keep the rest.
    const notMale = pool.find(v => !/\b(male|david|mark|guy|daniel|james)\b/i.test(v.name));
    return notMale || pool[0] || null;
  };

  const speakText = (text: string) => {
    if (typeof window === 'undefined' || !window.speechSynthesis) return;
    window.speechSynthesis.cancel();
    const cleanMsg = text.replace(/^\[[a-z]{2}-[A-Z]{2}\]\s*/i, '');
    const utterance = new SpeechSynthesisUtterance(cleanMsg);
    const voice = pickFemaleVoice();
    if (voice) utterance.voice = voice;
    utterance.rate = 1.15;
    utterance.pitch = 1.08;
    window.speechSynthesis.speak(utterance);
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

  const handleStartUploadedVideo = async () => {
    setUploadMessage({ text: '', type: '' });
    if (!selectedUploadedVideo) {
      setUploadMessage({ text: 'No uploaded video selected.', type: 'error' });
      return;
    }
    try {
      const res = await fetch(`${BACKEND_URL}/api/start_uploaded_video`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ filename: selectedUploadedVideo })
      });
      const data = await res.json();
      if (data.success) {
        setUploadMessage({ text: `Playing ${selectedUploadedVideo} -- full video runs through to the end, then Kickoff Triage picks up the final result.`, type: 'success' });
        // Force video feed img to reconnect (backend reset kills old MJPEG stream)
        setTimeout(() => setVideoFeedKey(k => k + 1), 400);
      } else {
        setUploadMessage({ text: data.error || 'Failed to start video.', type: 'error' });
      }
    } catch (e) {
      setUploadMessage({ text: 'Error starting uploaded video.', type: 'error' });
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
        if (data.mode === 'video') {
          // Videos are stored, not auto-played -- switch the Optical Device
          // to "Uploaded Video" mode and pre-select this file so the next
          // step is just: type a name, hit Play.
          await fetchUploadedVideos();
          setDeviceMode('upload');
          setSelectedUploadedVideo(data.filename || '');
          setUploadMessage({ text: `Saved: ${data.filename}. Select it under Optical Device -> Uploaded Video, then press Play.`, type: 'success' });
        } else {
          setUploadMessage({ text: `Media loaded: ${data.message}`, type: 'success' });
        }
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
    if (triageInFlightRef.current) return;
    triageInFlightRef.current = true;
    setIsTriageRunning(true);
    setLastTriageResult(null);
    try {
      // Always read from the current backend session state (which is
      // populated identically whether the source was a live webcam or an
      // uploaded video already played through the pipeline via /upload —
      // see agents/tools.py::get_vitals_dict's 'live' branch). There is no
      // UI path where this button has an actual File object to attach, so
      // the old FormData fallback below always 400'd once metrics.status
      // reached VIDEO_ENDED (it only appended patient_name, never a video
      // field) — meaning "upload, let it finish, then click Kickoff" was
      // silently broken. Sending 'live' unconditionally fixes that; a
      // truly empty session (no upload, no webcam) still gets a clear
      // WEBCAM_NO_DATA error from the backend instead of a misleading 400.
      const nameToSubmit = customName || patientName || '';
      const res = await fetch(`${BACKEND_URL}/api/triage_run`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ source: 'live', patient_name: nameToSubmit })
      });

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
      triageInFlightRef.current = false;
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

    // If the message names a patient already in the queue, auto-select their
    // record in the side panel so the clinician sees the same context ARIA
    // already has (the backend injects the full patient DB into every call).
    const mentioned = triageQueue.find(p => {
      const name = (p.name || '').trim().toLowerCase();
      return name.length > 2 && chatInput.toLowerCase().includes(name);
    });
    if (mentioned) setSelectedPatientId(mentioned.id);

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
        if (ttsEnabled) speakText(data.response);
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
      case 1: return 'bg-red-500/10 border-red-500/30 text-red-400';
      case 2: return 'bg-orange-500/10 border-orange-500/30 text-orange-400';
      case 3: return 'bg-yellow-500/10 border-yellow-500/30 text-yellow-400';
      case 4: return 'bg-emerald-500/10 border-emerald-500/30 text-emerald-400';
      case 5: return 'bg-cyan-500/10 border-cyan-500/30 text-cyan-400';
      default: return 'bg-white/5 border-white/10 text-zinc-300';
    }
  };

  return (
    <div className={`flex-1 font-sans min-h-screen antialiased selection:bg-cyan-500 selection:text-black transition-colors duration-200 ${
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
          box-shadow: 0 1px 3px rgba(0, 0, 0, 0.4);
        }
        /* Custom scrollbar for internal scroll panels (queue list, chat,
           agent console) — replaces the default OS/browser scrollbar, which
           looks out of place against this dark, rounded-card design. */
        .custom-scroll {
          scrollbar-width: thin;
          scrollbar-color: rgba(255, 255, 255, 0.14) transparent;
        }
        .custom-scroll::-webkit-scrollbar {
          width: 7px;
          height: 7px;
        }
        .custom-scroll::-webkit-scrollbar-track {
          background: transparent;
        }
        .custom-scroll::-webkit-scrollbar-thumb {
          background: rgba(255, 255, 255, 0.14);
          border-radius: 999px;
        }
        .custom-scroll::-webkit-scrollbar-thumb:hover {
          background: rgba(255, 255, 255, 0.26);
        }
        .light-theme .custom-scroll {
          scrollbar-color: rgba(15, 23, 42, 0.18) transparent;
        }
        .light-theme .custom-scroll::-webkit-scrollbar-thumb {
          background: rgba(15, 23, 42, 0.18);
        }
        .light-theme .custom-scroll::-webkit-scrollbar-thumb:hover {
          background: rgba(15, 23, 42, 0.32);
        }
        .shadow-glow-red {
          box-shadow: 0 1px 3px rgba(0, 0, 0, 0.4);
        }

        /* ── LIGHT THEME STYLES ──
           Wrapped in @layer utilities to match the layer Tailwind's own
           utility classes live in. Per the CSS Cascade Layers spec,
           !important declarations in *unlayered* CSS have LOWER priority
           than !important (or even normal) declarations inside a named
           layer — the importance ordering across layers is reversed
           relative to normal declarations. A plain unlayered <style>
           override, no matter how !important, could never beat a
           Tailwind utility for this reason. Placing these in the same
           layer makes it a normal same-layer, same-specificity contest,
           which source order (these rules come later) resolves correctly. */
        @layer utilities {
        .light-theme {
          background-color: #f8fafc !important;
          color: #0f172a !important;
        }
        .light-theme aside {
          background-color: rgba(255, 255, 255, 0.95) !important;
          border-color: #e2e8f0 !important;
          box-shadow: 4px 0 6px -1px rgba(0, 0, 0, 0.05) !important;
        }
        .light-theme aside h1, .light-theme aside span {
          color: #0f172a !important;
        }
        .light-theme aside p {
          color: #64748b !important;
        }
        /* Sidebar navigation */
        .light-theme aside nav button {
          color: #64748b !important;
        }
        .light-theme aside nav button:hover {
          background-color: #f1f5f9 !important;
          color: #0f172a !important;
        }
        /* Cards — substring matches catch every opacity/alpha-hex variant in one rule,
           instead of an exact-class list that silently misses new variants. */
        .light-theme [class*="bg-[#090e1e"] {
          background-color: rgba(255, 255, 255, 0.9) !important;
          border-color: #e2e8f0 !important;
          box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.05), 0 4px 6px -4px rgba(0, 0, 0, 0.05) !important;
        }
        /* Internal blocks (near-black backgrounds: consoles, waveforms, feeds) */
        .light-theme [class*="bg-[#02040a"],
        .light-theme [class*="bg-[#030712"],
        .light-theme [class*="bg-[#060813"],
        .light-theme [class*="bg-[#030612"] {
          background-color: #f1f5f9 !important;
          border-color: #e2e8f0 !important;
          color: #0f172a !important;
        }
        .light-theme [class*="bg-[#070b19"] {
          background-color: #ffffff !important;
          border-color: #e2e8f0 !important;
        }
        /* Agent-tab active highlight (was a dark indigo tint, unreadable on white) */
        .light-theme [class*="bg-[#0e1326"] {
          background-color: #e0f2fe !important;
        }
        /* Disabled gradient buttons (e.g. Kickoff Triage Crew when offline) rendered
           near-black-on-black in light mode; force a visibly muted, readable state. */
        .light-theme button:disabled {
          background-image: none !important;
          background-color: #e2e8f0 !important;
          color: #94a3b8 !important;
          opacity: 1 !important;
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

        /* ── Newer "professional" surfaces (sidebar cards, ARIA, Triage
           Dispatch, Clinical Crew) use a white/[x]-opacity + bg-[#0d1117]/
           bg-[#080a0f] vocabulary that didn't exist when the rules above
           were written, so light mode rendered them invisible (white-on-
           white) or washed-out gray. Same substring-selector approach,
           extended to cover these. ── */
        .light-theme [class*="bg-[#0d1117"] {
          background-color: rgba(255, 255, 255, 0.95) !important;
          border-color: #e2e8f0 !important;
          box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05) !important;
        }
        .light-theme [class*="bg-[#080a0f"] {
          background-color: #f1f5f9 !important;
          border-color: #e2e8f0 !important;
        }
        .light-theme [class*="bg-white/["] {
          background-color: #eef2f6 !important;
        }
        .light-theme [class*="border-white/"] {
          border-color: #dde3ea !important;
        }
        /* Text/icon grays tuned for a dark background read as too light on
           white — remap the ones the new sections actually use. */
        .light-theme .text-zinc-600,
        .light-theme .text-zinc-650,
        .light-theme .text-zinc-700 {
          color: #475569 !important;
        }
        /* Native <option> popups keep their own explicit dark colors
           (set inline in the JSX) even in light mode, since the closed
           <select> box is already handled above and native popup styling
           can't inherit Tailwind text-color the same way. */
        }
      `}</style>

      {/* 1 &amp; 2. Sidebar Navigation (collapsible) */}
      <div className="flex">
        <aside className={`sticky top-0 h-screen z-50 bg-[#070b19]/95 backdrop-blur-xl border-r border-zinc-800/80 shadow-xl flex flex-col flex-shrink-0 overflow-hidden transition-[width] duration-200 ease-out ${sidebarCollapsed ? 'w-[76px]' : 'w-64'}`}>

          {/* Logo */}
          <div className="flex items-center gap-3 px-4 py-5 border-b border-zinc-800/60">
            <div className="w-10 h-10 rounded-xl bg-zinc-950 flex items-center justify-center shadow-lg shadow-cyan-500/20 relative overflow-hidden border border-zinc-800 flex-shrink-0">
              <img src="/vital_logo.png" alt="VITAL Logo" className="w-full h-full object-cover" />
            </div>
            <div className={`overflow-hidden transition-opacity duration-150 whitespace-nowrap ${sidebarCollapsed ? 'opacity-0' : 'opacity-100'}`}>
              <h1 className="text-lg font-extrabold tracking-tight text-white leading-tight">VITAL</h1>
              <p className="text-[8.5px] text-zinc-500 uppercase tracking-widest font-bold mt-0.5 leading-tight">Vision-Based Triage &amp; Lifesign Analytics</p>
            </div>
          </div>

          {/* Status block: ARIA, System, Queue, Theme */}
          <div className="px-3 py-4 border-b border-zinc-800/60 flex flex-col gap-2">
            <div className={`flex items-center gap-2 text-xs bg-[#0b1022] border border-indigo-800/40 rounded-lg px-3 py-2 ${sidebarCollapsed ? 'justify-center' : ''}`} title="ARIA Online">
              <span className="w-2 h-2 rounded-full bg-indigo-400 animate-pulse flex-shrink-0"></span>
              <span className={`text-indigo-400 font-black uppercase tracking-widest text-[10px] overflow-hidden transition-opacity duration-150 whitespace-nowrap ${sidebarCollapsed ? 'w-0 opacity-0' : 'w-auto opacity-100'}`}>ARIA Online</span>
            </div>

            <div className={`flex items-center gap-2 text-xs bg-[#0b1022] border border-zinc-800 rounded-lg px-3 py-2 ${sidebarCollapsed ? 'justify-center' : ''}`} title={backendOnline ? 'System Online' : 'System Offline'}>
              {backendOnline ? (
                <>
                  <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse shadow-[0_0_8px_rgba(52,211,153,0.5)] flex-shrink-0"></span>
                  <span className={`font-extrabold text-emerald-400 text-[10px] uppercase tracking-widest overflow-hidden transition-opacity duration-150 whitespace-nowrap ${sidebarCollapsed ? 'w-0 opacity-0' : 'w-auto opacity-100'}`}>System Online</span>
                </>
              ) : (
                <>
                  <span className="w-2 h-2 rounded-full bg-rose-500 flex-shrink-0"></span>
                  <span className={`font-extrabold text-rose-500 text-[10px] uppercase tracking-widest overflow-hidden transition-opacity duration-150 whitespace-nowrap ${sidebarCollapsed ? 'w-0 opacity-0' : 'w-auto opacity-100'}`}>System Offline</span>
                </>
              )}
            </div>

            <div className={`flex items-center gap-2 text-xs bg-[#0b1022] border border-zinc-800 rounded-lg px-3 py-2 font-mono ${sidebarCollapsed ? 'justify-center' : ''}`} title="Triage Queue">
              <span className="text-zinc-500 uppercase tracking-wider text-[9px] font-bold flex-shrink-0">{sidebarCollapsed ? 'Q' : 'Queue'}</span>
              <span className="font-black text-cyan-400 text-sm">{triageQueue.length}</span>
            </div>

            {/* Theme Toggle Button */}
            <button
              onClick={() => setTheme(prev => prev === 'dark' ? 'light' : 'dark')}
              className={`flex items-center gap-2 p-2 bg-[#0b1022] border border-zinc-800 rounded-lg hover:bg-zinc-800/50 hover:border-zinc-700 transition-colors ${sidebarCollapsed ? 'justify-center' : ''}`}
              title={`Switch to ${theme === 'dark' ? 'Light' : 'Dark'} Mode`}
            >
              {theme === 'dark' ? (
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" className="text-amber-400 flex-shrink-0">
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
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" className="text-indigo-400 flex-shrink-0">
                  <path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"/>
                </svg>
              )}
              <span className={`text-[10px] uppercase font-bold text-zinc-400 tracking-widest overflow-hidden transition-opacity duration-150 whitespace-nowrap ${sidebarCollapsed ? 'w-0 opacity-0' : 'w-auto opacity-100'}`}>{theme === 'dark' ? 'Light Mode' : 'Dark Mode'}</span>
            </button>
          </div>

          {/* Section navigation */}
          <nav className="flex-1 flex flex-col gap-1.5 px-3 py-4 overflow-y-auto custom-scroll">
            <button
              onClick={() => goToTab('monitor')}
              title="Vitals Monitor"
              className={`flex items-center gap-3 rounded-lg px-3 py-3 transition-all text-xs font-black uppercase tracking-widest ${sidebarCollapsed ? 'justify-center' : ''} ${
                activeTab === 'monitor'
                  ? 'bg-cyan-500/10 text-cyan-400 border border-cyan-500/30 shadow-glow-cyan'
                  : 'text-zinc-500 hover:text-zinc-300 border border-transparent hover:bg-zinc-800/30'
              }`}
            >
              <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" className="flex-shrink-0"><polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/></svg>
              <span className={`overflow-hidden transition-opacity duration-150 whitespace-nowrap ${sidebarCollapsed ? 'w-0 opacity-0' : 'w-auto opacity-100'}`}>Vitals Monitor</span>
            </button>
            <button
              onClick={() => goToTab('queue')}
              title="Triage Dispatch"
              className={`flex items-center gap-3 rounded-lg px-3 py-3 transition-all text-xs font-black uppercase tracking-widest ${sidebarCollapsed ? 'justify-center' : ''} ${
                activeTab === 'queue'
                  ? 'bg-cyan-500/10 text-cyan-400 border border-cyan-500/30 shadow-glow-cyan'
                  : 'text-zinc-500 hover:text-zinc-300 border border-transparent hover:bg-zinc-800/30'
              }`}
            >
              <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" className="flex-shrink-0"><path d="M9 11l3 3L22 4"/><path d="M21 12v7a2 2 0 01-2 2H5a2 2 0 01-2-2V5a2 2 0 012-2h11"/></svg>
              <span className={`flex-1 text-left overflow-hidden transition-opacity duration-150 whitespace-nowrap ${sidebarCollapsed ? 'w-0 opacity-0' : 'w-auto opacity-100'}`}>Triage Dispatch</span>
              {triageQueue.length > 0 && (
                <span className="bg-cyan-500 text-[#020408] text-[9px] font-black px-1.5 py-0.5 rounded-full leading-none flex-shrink-0">{triageQueue.length}</span>
              )}
            </button>
            <button
              onClick={() => goToTab('crew')}
              title="Clinical Crew"
              className={`flex items-center gap-3 rounded-lg px-3 py-3 transition-all text-xs font-black uppercase tracking-widest ${sidebarCollapsed ? 'justify-center' : ''} ${
                activeTab === 'crew'
                  ? 'bg-cyan-500/10 text-cyan-400 border border-cyan-500/30 shadow-glow-cyan'
                  : 'text-zinc-500 hover:text-zinc-300 border border-transparent hover:bg-zinc-800/30'
              }`}
            >
              <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" className="flex-shrink-0"><circle cx="12" cy="8" r="4"/><path d="M4 20c0-4 3.6-7 8-7s8 3 8 7"/></svg>
              <span className={`overflow-hidden transition-opacity duration-150 whitespace-nowrap ${sidebarCollapsed ? 'w-0 opacity-0' : 'w-auto opacity-100'}`}>Clinical Crew</span>
            </button>
            <button
              onClick={() => goToTab('chat')}
              title="ARIA Assistant"
              className={`flex items-center gap-3 rounded-lg px-3 py-3 transition-all text-xs font-black uppercase tracking-widest ${sidebarCollapsed ? 'justify-center' : ''} ${
                activeTab === 'chat'
                  ? 'bg-indigo-500/10 text-indigo-400 border border-indigo-500/30 shadow-[0_0_15px_rgba(99,102,241,0.15)]'
                  : 'text-zinc-500 hover:text-zinc-300 border border-transparent hover:bg-zinc-800/30'
              }`}
            >
              <span className="w-2 h-2 rounded-full bg-indigo-400 animate-pulse flex-shrink-0"></span>
              <span className={`overflow-hidden transition-opacity duration-150 whitespace-nowrap ${sidebarCollapsed ? 'w-0 opacity-0' : 'w-auto opacity-100'}`}>ARIA Assistant</span>
            </button>
          </nav>

          {/* Collapse / expand toggle — bottom of sidebar */}
          <button
            onClick={() => setSidebarCollapsed(v => !v)}
            className={`flex items-center gap-2 mx-3 mb-3 px-3 py-2.5 rounded-lg bg-[#0b1022] border border-zinc-800 text-zinc-500 hover:text-cyan-400 hover:border-cyan-500/40 hover:bg-zinc-800/50 transition-colors ${sidebarCollapsed ? 'justify-center' : ''}`}
            title={sidebarCollapsed ? 'Expand sidebar' : 'Collapse sidebar'}
          >
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" className={`flex-shrink-0 transition-transform duration-200 ${sidebarCollapsed ? 'rotate-180' : ''}`}>
              <polyline points="15 18 9 12 15 6"/>
            </svg>
            <span className={`text-[10px] uppercase font-bold tracking-widest overflow-hidden transition-opacity duration-150 whitespace-nowrap ${sidebarCollapsed ? 'w-0 opacity-0' : 'w-auto opacity-100'}`}>Collapse</span>
          </button>
        </aside>

      {/* 3. Main Workspace Container */}
      <main className="flex-1 min-w-0 max-w-[1700px] mx-auto px-6 mt-6 pb-10">

        {/* VIEW 1: LIVE VITAL MONITOR */}
        {activeTab === 'monitor' && (
          <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-stretch">
            
            {/* Left Hand side: Camera Feed Controls (expanded to col-span-7) */}
            <div className="lg:col-span-7 flex flex-col gap-6">
              <div className="bg-[#0d1117]/70 border border-white/[0.06] rounded-2xl p-6 backdrop-blur-xl shadow-lg flex flex-col relative">

                <div className="flex items-center justify-between mb-4">
                  <h2 className="text-[11px] font-semibold text-zinc-400 uppercase tracking-wide flex items-center gap-2">
                    <span className="w-1.5 h-1.5 rounded-full bg-cyan-400/80"></span>
                    Intake Acquisition Feed
                  </h2>
                  {metrics.is_live && (
                    <span className="text-[10px] font-semibold text-rose-400 flex items-center gap-1.5">
                      <span className="w-1.5 h-1.5 rounded-full bg-rose-500 animate-pulse"></span>
                      Recording
                    </span>
                  )}
                </div>

                <div className="group relative min-h-[420px] bg-[#080a0f] border border-white/[0.06] rounded-xl overflow-hidden mb-4 flex items-center justify-center">

                  {backendOnline ? (
                    metrics.status === 'IMAGE_READY' || metrics.remark === 'IMAGE_DEMO' ? (
                      <img
                        src={`${BACKEND_URL}/image_feed`}
                        alt="Intake snap"
                        className="w-full h-full object-contain relative z-10"
                      />
                    ) : (
                      <img
                        key={videoFeedKey}
                        src={`${BACKEND_URL}/video_feed?t=${videoFeedKey}`}
                        alt="Medical Stream"
                        className="w-full h-full object-contain relative z-10"
                        onError={() => {
                          // Auto-retry connection after 1s if stream breaks
                          setTimeout(() => setVideoFeedKey(k => k + 1), 1000);
                        }}
                      />
                    )
                  ) : (
                    <div className="text-center p-6 max-w-[320px] relative z-10">
                      <div className="w-14 h-14 mx-auto mb-4 flex items-center justify-center rounded-full bg-white/[0.04] border border-white/[0.06]">
                        <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.6" className="text-zinc-500">
                          <path d="M1 5h4l2-3h6l2 3h4a2 2 0 0 1 2 2v11a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V7a2 2 0 0 1 2-2z" opacity="0.6"/>
                          <circle cx="12" cy="12" r="3.2"/>
                          <line x1="2" y1="20" x2="20" y2="2" stroke="currentColor" strokeWidth="1.6" className="text-zinc-500/70"/>
                        </svg>
                      </div>
                      <p className="text-sm font-semibold text-zinc-300">Intake hardware offline</p>
                      <p className="text-xs text-zinc-500 mt-2 leading-relaxed">Ensure the Python backend API server is running on port 5002 with required dependencies.</p>
                    </div>
                  )}

                  {/* Floating acquisition state tags */}
                  <div className="absolute top-3 left-3 flex flex-wrap gap-2 pointer-events-none z-10">
                    <span className={`text-[10px] font-semibold px-2.5 py-1 rounded-md border backdrop-blur-sm ${
                      metrics.status === 'OK' ? 'bg-emerald-500/10 text-emerald-400 border-emerald-500/20' :
                      metrics.status === 'CALIBRATING' ? 'bg-cyan-500/10 text-cyan-400 border-cyan-500/20' :
                      metrics.status === 'VIDEO_ENDED' ? 'bg-white/5 text-zinc-400 border-white/10' :
                      metrics.status === 'IMAGE_READY' ? 'bg-purple-500/10 text-purple-400 border-purple-500/20' :
                      'bg-white/5 text-zinc-500 border-white/10'
                    }`}>
                      {metrics.status}
                    </span>

                    {metrics.face_detected && (
                      <span className="text-[10px] bg-cyan-500/10 text-cyan-400 border border-cyan-500/20 px-2.5 py-1 rounded-md font-semibold backdrop-blur-sm">
                        Face detected
                      </span>
                    )}
                  </div>

                  {metrics.video_total_sec != null && (
                    <span
                      className="absolute bottom-3 right-3 text-[11px] font-mono font-semibold px-2.5 py-1 rounded-md bg-black/70 text-zinc-200 border border-white/10 backdrop-blur-sm opacity-0 group-hover:opacity-100 transition-opacity pointer-events-none z-20"
                      title="Uploaded video playback position"
                    >
                      {formatDuration(metrics.video_current_sec || 0)} / {formatDuration(metrics.video_total_sec)}
                      {metrics.status === 'VIDEO_ENDED' && ' (ended)'}
                    </span>
                  )}
                </div>

                {metrics.status === 'VIDEO_ENDED' && !metrics.is_live && (
                  <div className="mb-4 flex items-center justify-between gap-3 bg-amber-500/10 border border-amber-500/25 rounded-lg px-4 py-3">
                    <span className="text-xs font-semibold text-amber-300 flex items-center gap-2">
                      {isTriageRunning ? (
                        <>
                          <span className="w-3 h-3 rounded-full border-2 border-amber-400 border-t-transparent animate-spin"></span>
                          Video finished -- Triage Crew is running automatically on the full-video result...
                        </>
                      ) : (
                        <>Video finished -- Triage Crew ran automatically on the full-video result.</>
                      )}
                    </span>
                    <button
                      onClick={() => goToTab('crew')}
                      className="shrink-0 text-xs font-bold text-amber-200 bg-amber-500/15 hover:bg-amber-500/25 border border-amber-500/30 rounded-md px-3 py-1.5 transition-colors"
                    >
                      View in Clinical Crew &rarr;
                    </button>
                  </div>
                )}

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
                        value={deviceMode === 'upload' ? '__upload__' : (selectedCamera !== null ? selectedCamera : '')}
                        onChange={(e) => {
                          if (e.target.value === '__upload__') {
                            setDeviceMode('upload');
                          } else {
                            setDeviceMode('camera');
                            setSelectedCamera(Number(e.target.value));
                          }
                        }}
                        className="bg-[#02040a] border border-zinc-800 rounded-lg text-sm p-2.5 text-zinc-300 outline-none focus:border-cyan-500 transition-colors cursor-pointer"
                      >
                        {cameras.map((cam) => (
                          <option key={cam.index} value={cam.index} style={{ backgroundColor: '#111827', color: '#e5e7eb' }}>{cam.label}</option>
                        ))}
                        {cameras.length === 0 && <option value="" style={{ backgroundColor: '#111827', color: '#e5e7eb' }}>No hardware found</option>}
                        <option value="__upload__" style={{ backgroundColor: '#111827', color: '#e5e7eb' }}>Uploaded Video</option>
                      </select>
                    </div>

                    <div className="flex flex-col gap-1.5">
                      <label className="text-[10px] text-zinc-450 font-black uppercase tracking-wider">Interface controls</label>
                      <div className="flex gap-2">
                        <button
                          onClick={deviceMode === 'upload' ? handleStartUploadedVideo : handleStartWebcam}
                          className="flex-1 bg-cyan-500/10 hover:bg-cyan-500/20 text-cyan-400 border border-cyan-500/40 hover:border-cyan-500/60 font-bold text-xs py-2.5 px-3 rounded-lg transition-all shadow-glow-cyan"
                        >
                          {deviceMode === 'upload' ? 'Load Video' : 'Init Sensors'}
                        </button>
                        <button
                          onClick={metrics.is_live ? handleReleaseCamera : (deviceMode === 'upload' ? handleStartUploadedVideo : handleStartWebcam)}
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

                  {deviceMode === 'upload' && (
                    <div className="flex flex-col gap-1.5">
                      <label className="text-[10px] text-zinc-450 font-black uppercase tracking-wider">Uploaded Video (select which one to play)</label>
                      <select
                        value={selectedUploadedVideo}
                        onChange={(e) => setSelectedUploadedVideo(e.target.value)}
                        className="bg-[#02040a] border border-zinc-800 rounded-lg text-sm p-2.5 text-zinc-300 outline-none focus:border-cyan-500 transition-colors cursor-pointer"
                      >
                        {uploadedVideos.length === 0 && (
                          <option value="" style={{ backgroundColor: '#111827', color: '#e5e7eb' }}>No videos uploaded yet</option>
                        )}
                        {uploadedVideos.map((v) => (
                          <option key={v.filename} value={v.filename} style={{ backgroundColor: '#111827', color: '#e5e7eb' }}>
                            {v.filename} ({v.modified})
                          </option>
                        ))}
                      </select>
                    </div>
                  )}

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
              <div className="bg-[#0d1117]/70 border border-white/[0.06] rounded-2xl p-6 backdrop-blur-xl shadow-lg relative">
                <div className="flex items-center justify-between mb-4">
                  <h2 className="text-[11px] font-semibold text-zinc-400 uppercase tracking-wide flex items-center gap-2">
                    <span className="w-1.5 h-1.5 rounded-full bg-emerald-400/80"></span>
                    Optical Physiology Indicators
                  </h2>
                  {/* 30-sec session acquisition timer */}
                  <div className="flex items-center gap-2 ml-auto">
                    {metrics.calibration_done && sessionTimer > 0 && (
                      <div className="flex items-center gap-2 bg-white/[0.03] border border-white/[0.06] rounded-lg px-3 py-1.5">
                        <svg width="26" height="26" viewBox="0 0 36 36" className="-rotate-90">
                          <circle cx="18" cy="18" r="14" fill="none" stroke="rgba(255,255,255,0.08)" strokeWidth="3"/>
                          <circle cx="18" cy="18" r="14" fill="none" stroke="#22d3ee" strokeWidth="3"
                            strokeDasharray={`${2 * Math.PI * 14}`}
                            strokeDashoffset={`${2 * Math.PI * 14 * (1 - sessionTimer / 30)}`}
                            strokeLinecap="round" style={{transition: 'stroke-dashoffset 1s linear'}}/>
                          <text x="18" y="23" textAnchor="middle" fill="#22d3ee" fontSize="10" fontWeight="600"
                            style={{transform: 'rotate(90deg)', transformOrigin: '18px 18px'}}>{sessionTimer}</text>
                        </svg>
                        <div className="text-right">
                          <div className="text-[9px] text-zinc-500 font-medium uppercase tracking-wide">Scan window</div>
                          <div className="text-[11px] text-cyan-400 font-semibold">{sessionTimer}s remaining</div>
                        </div>
                      </div>
                    )}
                    {sessionTimer === 0 && !reportLoading && sessionReport && (
                      <div className="text-[10px] bg-emerald-500/10 border border-emerald-500/20 text-emerald-400 px-3 py-1.5 rounded-lg font-semibold flex items-center gap-1.5">
                        <span className="w-1.5 h-1.5 rounded-full bg-emerald-400"></span>
                        Report ready
                      </div>
                    )}
                    {reportLoading && (
                      <div className="text-[10px] text-cyan-400 font-semibold flex items-center gap-1.5">
                        <span className="w-1.5 h-1.5 rounded-full bg-cyan-400 animate-pulse"></span>
                        Compiling report...
                      </div>
                    )}
                  </div>
                </div>

                <div className="grid grid-cols-1 md:grid-cols-3 gap-4">

                  {/* HEART RATE */}
                  <div className={`bg-white/[0.02] border rounded-xl p-5 flex flex-col justify-between hover:border-white/20 transition-all ${
                    metrics.classification === 'TACHYCARDIA' ? 'border-rose-500/30' :
                    metrics.classification === 'NORMAL' ? 'border-emerald-500/20' :
                    'border-white/[0.06]'
                  }`}>
                    <div className="flex items-center justify-between">
                      <span className="text-[10px] text-zinc-400 font-semibold uppercase tracking-wide">Heart Rate</span>
                      <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" className="text-rose-400/70">
                        <path d="M12 21s-7.1-4.4-9.6-8.7C.6 8.9 2 5.2 5.6 5.2c2 0 3.4 1.2 4.1 2.2.7-1 2.1-2.2 4.1-2.2 3.5 0 5 3.7 3.2 7.1C19.1 16.6 12 21 12 21z"/>
                      </svg>
                    </div>
                    <div className="my-3 flex items-baseline gap-2">
                      <span className="text-4xl font-semibold text-white tracking-tight leading-none font-mono">
                        {metrics.bpm > 0 ? metrics.bpm : '--'}
                      </span>
                      <span className="text-[11px] text-zinc-500 font-medium uppercase tracking-wide">BPM</span>
                    </div>
                    <div className="flex items-center justify-between border-t border-white/[0.06] pt-2">
                      <span className={`text-[10px] px-2 py-0.5 rounded font-semibold ${
                        metrics.classification === 'NORMAL' ? 'bg-emerald-500/10 text-emerald-400' :
                        metrics.classification === 'TACHYCARDIA' ? 'bg-rose-500/10 text-rose-400' :
                        metrics.classification === 'BRADYCARDIA' ? 'bg-cyan-500/10 text-cyan-400' :
                        'bg-white/5 text-zinc-500'
                      }`}>
                        {metrics.classification}
                      </span>
                      <span className="text-[10px] text-zinc-500 font-mono">Conf {typeof metrics.confidence === 'number' ? metrics.confidence.toFixed(1) : metrics.confidence}%</span>
                    </div>
                  </div>

                  {/* RESPIRATORY RATE */}
                  <div className={`bg-white/[0.02] border rounded-xl p-5 flex flex-col justify-between hover:border-white/20 transition-all ${
                    metrics.rr_classification === 'TACHYPNEA' ? 'border-rose-500/30' :
                    metrics.rr_classification === 'NORMAL' ? 'border-emerald-500/20' :
                    'border-white/[0.06]'
                  }`}>
                    <div className="flex items-center justify-between">
                      <span className="text-[10px] text-zinc-400 font-semibold uppercase tracking-wide">Respiration</span>
                      <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" className="text-cyan-400/70">
                        <path d="M9.6 4.6a2 2 0 1 1 1.4 3.4H2m10.6 11.4a2 2 0 1 0 1.4-3.4H2m15.7-8.3a2.5 2.5 0 1 1 1.8 4.3H2"/>
                      </svg>
                    </div>
                    <div className="my-3 flex items-baseline gap-2 overflow-hidden">
                      <span className="text-4xl font-semibold text-white tracking-tight leading-none font-mono truncate">
                        {metrics.rr > 0 ? Number(metrics.rr).toFixed(1) : '--'}
                      </span>
                      <span className="text-[11px] text-zinc-500 font-medium uppercase tracking-wide shrink-0">B/min</span>
                    </div>
                    <div className="flex items-center justify-between border-t border-white/[0.06] pt-2">
                      <span className={`text-[10px] px-2 py-0.5 rounded font-semibold ${
                        metrics.rr_classification === 'NORMAL' ? 'bg-emerald-500/10 text-emerald-400' :
                        metrics.rr_classification === 'TACHYPNEA' ? 'bg-rose-500/10 text-rose-400' :
                        metrics.rr_classification === 'BRADYPNEA' ? 'bg-cyan-500/10 text-cyan-400' :
                        'bg-white/5 text-zinc-500'
                      }`}>
                        {metrics.rr_classification}
                      </span>
                      <span className="text-[10px] text-zinc-500 font-mono">Conf {typeof metrics.rr_confidence === 'number' ? Number(metrics.rr_confidence).toFixed(1) : metrics.rr_confidence}%</span>
                    </div>
                  </div>

                  {/* STRESS INDEX */}
                  <div className={`bg-white/[0.02] border rounded-xl p-5 flex flex-col justify-between hover:border-white/20 transition-all ${
                    metrics.stress_index > 150 ? 'border-rose-500/30' :
                    metrics.stress_index > 80 ? 'border-amber-500/25' :
                    'border-white/[0.06]'
                  }`}>
                    <div className="flex items-center justify-between">
                      <span className="text-[10px] text-zinc-400 font-semibold uppercase tracking-wide">Stress Index</span>
                      <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" className="text-amber-400/70">
                        <path d="M13 2 3 14h9l-1 8 10-12h-9l1-8z"/>
                      </svg>
                    </div>
                    <div className="my-3 flex items-baseline gap-2 overflow-hidden">
                      <span className={`font-semibold text-white tracking-tight leading-none font-mono ${
                        metrics.stress_index > 999 ? 'text-2xl' : 'text-4xl'
                      }`}>
                        {metrics.stress_index > 0 ? Number(metrics.stress_index).toFixed(0) : '--'}
                      </span>
                      <span className="text-[11px] text-zinc-500 font-medium uppercase tracking-wide shrink-0">Index</span>
                    </div>
                    <div className="flex items-center justify-between border-t border-white/[0.06] pt-2">
                      <span className={`text-[10px] px-2 py-0.5 rounded font-semibold ${
                        metrics.stress_index > 150 ? 'bg-rose-500/10 text-rose-400' :
                        metrics.stress_index > 80 ? 'bg-amber-500/10 text-amber-400' :
                        metrics.stress_index > 0 ? 'bg-emerald-500/10 text-emerald-400' :
                        'bg-white/5 text-zinc-500'
                      }`}>
                        {metrics.stress_index > 150 ? 'Critical' : metrics.stress_index > 80 ? 'Elevated' : metrics.stress_index > 0 ? 'Optimal' : '--'}
                      </span>
                      <span className="text-[10px] text-zinc-500 font-mono">HRV {typeof metrics.hrv === 'number' ? Number(metrics.hrv).toFixed(0) : '--'}ms</span>
                    </div>
                  </div>

                </div>

                <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mt-4">
                  {/* SIGNAL QUALITY & SNR */}
                  <div className="bg-white/[0.02] border border-white/[0.06] rounded-xl p-4 flex items-center justify-between hover:border-white/20 transition-all">
                    <div>
                      <span className="text-[10px] text-zinc-500 font-semibold uppercase tracking-wide">Signal SNR</span>
                      <div className="text-xl font-semibold text-white font-mono mt-1">{metrics.snr_db ? `${metrics.snr_db.toFixed(1)} dB` : '--'}</div>
                    </div>
                    <div className="text-right">
                      <span className="text-[10px] text-zinc-500 font-semibold uppercase tracking-wide">Stability</span>
                      <div className="text-sm font-medium text-cyan-400 font-mono mt-1">{metrics.stability_indicator} ({metrics.stability.toFixed(1)} bpm)</div>
                    </div>
                  </div>

                  {/* LIGHT & MOTION */}
                  <div className="bg-white/[0.02] border border-white/[0.06] rounded-xl p-4 flex items-center justify-between hover:border-white/20 transition-all">
                    <div>
                      <span className="text-[10px] text-zinc-500 font-semibold uppercase tracking-wide">Acquisition Environment</span>
                      <div className="text-sm font-medium text-zinc-300 mt-1 flex flex-col gap-0.5">
                        <span>Luminance: <strong className="text-white font-mono">{metrics.estimated_lux} lux</strong></span>
                        <span>Motion: <strong className="text-white font-mono">{metrics.motion_delta.toFixed(1)}</strong></span>
                      </div>
                    </div>
                    <div className="text-[10px] text-zinc-550 text-right leading-relaxed font-medium">
                      <div>Limit: &gt;100 lux</div>
                      <div>Motion: &lt;15.0</div>
                    </div>
                  </div>
                </div>

                {metrics.warnings && metrics.warnings.length > 0 && (
                  <div className="mt-4 p-4 bg-rose-500/[0.06] border border-rose-500/20 rounded-xl flex flex-col gap-1.5">
                    <div className="text-xs font-semibold text-rose-400 flex items-center gap-1.5">
                      ⚠ Telemetry acquisition warnings
                    </div>
                    <ul className="list-disc list-inside text-xs text-rose-300/90 font-medium leading-relaxed">
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
                      <button
                        onClick={() => setSessionReport(null)}
                        title="Dismiss this report (does not start a new scan)"
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
          <div className="bg-[#0d1117]/70 border border-white/[0.06] rounded-2xl p-6 backdrop-blur-xl shadow-lg flex flex-col h-[calc(100vh-48px)] relative">

            <div className="flex items-center justify-between border-b border-white/[0.06] pb-4 mb-5 flex-shrink-0">
              <div>
                <h2 className="text-base font-semibold text-white flex items-center gap-2">
                  <span className="w-1.5 h-1.5 rounded-full bg-cyan-400"></span>
                  Central Dispatch Triage Queue
                </h2>
                <p className="text-[11px] text-zinc-500 font-medium mt-1">Dynamically sorted by acuity — ESI 1 &amp; 2 prioritize to the top</p>
              </div>
              <button
                onClick={handleClearQueue}
                className="text-rose-400 hover:text-rose-300 hover:bg-rose-500/10 border border-rose-500/20 text-xs font-semibold py-2 px-5 rounded-lg transition-colors"
              >
                Flush queue
              </button>
            </div>

            <div className="flex-1 overflow-y-auto custom-scroll flex flex-col gap-3 pr-1 min-h-0">
              {triageQueue.length === 0 ? (
                <div className="h-full flex flex-col items-center justify-center text-center py-24">
                  <div className="w-14 h-14 mx-auto mb-4 flex items-center justify-center rounded-full bg-white/[0.04] border border-white/[0.06]">
                    <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.6" className="text-zinc-500"><path d="M9 2h6a1 1 0 011 1v2H8V3a1 1 0 011-1z"/><path d="M4 7a2 2 0 012-2h12a2 2 0 012 2v13a2 2 0 01-2 2H6a2 2 0 01-2-2V7z"/><path d="M9 12h6M9 16h6"/></svg>
                  </div>
                  <p className="text-sm font-semibold text-zinc-300">Clinical queue empty</p>
                  <p className="text-[11px] text-zinc-500 mt-1.5">Processed ESI patient records will compile here.</p>
                </div>
              ) : (
                triageQueue.map((patient) => {
                  const isExpanded = expandedPatientId === patient.id;
                  return (
                  <div
                    key={patient.id}
                    className={`border rounded-xl p-5 bg-white/[0.02] hover:bg-white/[0.04] transition-all flex flex-col relative flex-shrink-0 ${
                      patient.is_shock ? 'border-rose-500/30' : 'border-white/[0.06] hover:border-white/20'
                    }`}
                  >
                    {/* Urgency Sidebar Indicator */}
                    <div className={`absolute left-0 top-0 bottom-0 w-1 ${
                      patient.esi_level === 1 ? 'bg-red-500' :
                      patient.esi_level === 2 ? 'bg-orange-500' :
                      patient.esi_level === 3 ? 'bg-yellow-500' :
                      patient.esi_level === 4 ? 'bg-emerald-500' :
                      'bg-cyan-500'
                    }`}></div>

                    <div className="flex flex-col md:flex-row md:items-center justify-between gap-6">
                      <div className="flex-1 pl-3">
                        <div className="flex flex-wrap items-center gap-3">
                          <span className="font-semibold text-white text-base">{patient.name}</span>
                          <span className="text-xs font-mono text-zinc-500">{patient.timestamp}</span>
                          {patient.is_shock && (
                            <span className="text-[10px] bg-rose-500/10 text-rose-400 border border-rose-500/20 px-2.5 py-0.5 rounded font-semibold">
                              Compensated shock alert
                            </span>
                          )}
                        </div>

                        <div className="mt-2 text-sm text-zinc-300 leading-relaxed max-w-[1200px] font-medium">
                          {patient.triage_summary}
                        </div>

                        <div className="mt-3.5 flex flex-wrap items-center gap-5 text-[10px] text-zinc-500 font-medium">
                          <span>Record ID: <strong className="text-zinc-300 font-mono font-medium">{patient.id}</strong></span>
                          <span>Source: <strong className="text-zinc-300 font-mono font-medium">{patient.video_path}</strong></span>
                          <span>Diagnosis: <strong className="text-indigo-400 font-medium">{patient.primary_diagnosis}</strong></span>
                        </div>
                      </div>

                      <div className="flex flex-row md:flex-col items-center gap-3 self-start md:self-auto pl-3 md:pl-0">
                        <div className="flex items-center gap-2">
                          <div className={`border rounded-lg py-2 px-4 text-center min-w-[90px] ${getEsiClass(patient.esi_level)}`}>
                            <div className="text-[9px] uppercase font-medium tracking-wide opacity-80">ESI Level</div>
                            <div className="text-xl font-semibold font-mono leading-none mt-1">{patient.esi_level}</div>
                          </div>
                          <button
                            onClick={() => setExpandedPatientId(isExpanded ? null : patient.id)}
                            title={isExpanded ? 'Hide vitals' : 'Show vitals'}
                            className={`p-1.5 rounded-lg border transition-colors ${
                              isExpanded ? 'bg-cyan-500/10 border-cyan-500/30 text-cyan-400' : 'bg-white/[0.02] border-white/[0.08] text-zinc-500 hover:text-zinc-300 hover:bg-white/[0.05]'
                            }`}
                          >
                            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" className={`transition-transform duration-200 ${isExpanded ? 'rotate-180' : ''}`}>
                              <polyline points="6 9 12 15 18 9"/>
                            </svg>
                          </button>
                        </div>
                        <div className="bg-white/[0.03] border border-white/[0.08] text-zinc-300 font-mono text-[10px] px-3 py-1.5 rounded text-center min-w-[90px] font-medium">
                          Score: {patient.priority_score}
                        </div>
                      </div>
                    </div>

                    {isExpanded && (
                      <div className="mt-4 pt-4 pl-3 border-t border-white/[0.06] grid grid-cols-2 md:grid-cols-4 gap-3">
                        <div className="bg-white/[0.02] border border-white/[0.06] rounded-lg p-3 text-center">
                          <div className="text-[9px] text-zinc-500 font-semibold uppercase tracking-wide">Heart Rate</div>
                          <div className="text-lg font-semibold text-white font-mono mt-1">{patient.heart_rate != null ? `${Number(patient.heart_rate).toFixed(1)}` : '--'} <span className="text-[10px] text-zinc-500 font-medium">bpm</span></div>
                        </div>
                        <div className="bg-white/[0.02] border border-white/[0.06] rounded-lg p-3 text-center">
                          <div className="text-[9px] text-zinc-500 font-semibold uppercase tracking-wide">Respiration</div>
                          <div className="text-lg font-semibold text-white font-mono mt-1">{patient.respiration != null ? `${Number(patient.respiration).toFixed(1)}` : '--'} <span className="text-[10px] text-zinc-500 font-medium">br/min</span></div>
                        </div>
                        <div className="bg-white/[0.02] border border-white/[0.06] rounded-lg p-3 text-center">
                          <div className="text-[9px] text-zinc-500 font-semibold uppercase tracking-wide">HRV</div>
                          <div className="text-lg font-semibold text-white font-mono mt-1">{patient.hrv != null ? `${Number(patient.hrv).toFixed(1)}` : '--'} <span className="text-[10px] text-zinc-500 font-medium">ms</span></div>
                        </div>
                        <div className="bg-white/[0.02] border border-white/[0.06] rounded-lg p-3 text-center">
                          <div className="text-[9px] text-zinc-500 font-semibold uppercase tracking-wide">Stress Index</div>
                          <div className="text-lg font-semibold text-white font-mono mt-1">{patient.stress_index != null ? `${Number(patient.stress_index).toFixed(0)}` : '--'}</div>
                        </div>
                      </div>
                    )}
                  </div>
                  );
                })
              )}
            </div>

          </div>
        )}

        {/* VIEW 3: AGENT CREW PANEL */}
        {activeTab === 'crew' && (
          <div className="bg-[#0d1117]/70 border border-white/[0.06] rounded-2xl p-6 backdrop-blur-xl shadow-lg flex flex-col h-[calc(100vh-48px)] relative">

            <div className="flex items-center justify-between border-b border-white/[0.06] pb-4 mb-5 flex-shrink-0">
              <div>
                <h2 className="text-base font-semibold text-white flex items-center gap-2">
                  <span className="w-1.5 h-1.5 rounded-full bg-purple-400"></span>
                  Multi-Agent Clinical Intelligence Crew
                </h2>
                <p className="text-[11px] text-zinc-500 font-medium mt-1">Execution logs of the 3-agent decoupled triage pipeline</p>
              </div>

              <button
                onClick={() => handleRunTriage()}
                disabled={isTriageRunning || metrics.status === 'DISCONNECTED' || triageAutoRan}
                title={triageAutoRan ? 'Triage already ran automatically for this captured session' : undefined}
                className="bg-indigo-500/10 hover:bg-indigo-500/20 disabled:bg-white/[0.03] disabled:text-zinc-500 disabled:cursor-not-allowed text-indigo-400 border border-indigo-500/25 disabled:border-white/[0.06] font-semibold text-sm py-2.5 px-5 rounded-xl transition-colors flex items-center gap-2.5"
              >
                {isTriageRunning ? (
                  <>
                    <span className="w-3.5 h-3.5 rounded-full border-2 border-indigo-400 border-t-transparent animate-spin"></span>
                    Negotiating...
                  </>
                ) : triageAutoRan ? (
                  <>Already dispatched</>
                ) : (
                  <>Kickoff Triage Crew</>
                )}
              </button>
            </div>

            {/* Agent selecting buttons */}
            <div className="flex border-b border-white/[0.06] mb-5 flex-shrink-0">
              <button
                onClick={() => setActiveAgentTab('perception')}
                className={`flex-1 py-3 text-xs font-semibold transition-all border-b-2 ${
                  activeAgentTab === 'perception' ? 'text-cyan-400 border-cyan-400' : 'text-zinc-500 border-transparent hover:text-zinc-300'
                }`}
              >
                Perception Agent
              </button>
              <button
                onClick={() => setActiveAgentTab('diagnostic')}
                className={`flex-1 py-3 text-xs font-semibold transition-all border-b-2 ${
                  activeAgentTab === 'diagnostic' ? 'text-amber-400 border-amber-400' : 'text-zinc-500 border-transparent hover:text-zinc-300'
                }`}
              >
                Diagnostic Agent
              </button>
              <button
                onClick={() => setActiveAgentTab('coordinator')}
                className={`flex-1 py-3 text-xs font-semibold transition-all border-b-2 ${
                  activeAgentTab === 'coordinator' ? 'text-indigo-400 border-indigo-400' : 'text-zinc-500 border-transparent hover:text-zinc-300'
                }`}
              >
                Coordinator Agent
              </button>
            </div>

            {/* Main console screen */}
            <div className="bg-white/[0.02] border border-white/[0.06] rounded-xl p-5 flex-1 overflow-y-auto custom-scroll font-mono text-sm leading-relaxed text-zinc-300 min-h-0">
              {isTriageRunning && (
                <div className="h-full flex flex-col items-center justify-center text-zinc-500 py-16 gap-4">
                  <div className="w-8 h-8 rounded-full border-2 border-cyan-400 border-t-transparent animate-spin"></div>
                  <div className="text-center font-sans">
                    <p className="text-sm font-semibold text-zinc-200">Multi-agent negotiation active</p>
                    <p className="text-xs text-zinc-500 mt-2 max-w-[340px] leading-relaxed font-medium">Consolidating rPPG bio-telemetry, analyzing shock state indicators, and prioritizing patients...</p>
                  </div>
                </div>
              )}

              {!isTriageRunning && !lastTriageResult && (
                <div className="h-full flex items-center justify-center text-center py-20 font-sans text-xs font-medium text-zinc-500">
                  <div>
                    <p>Sensor pipeline idle.</p>
                    <p className="text-zinc-600 mt-1.5 text-[10px] font-mono">Upload a record or start live webcam and trigger the Triage Crew.</p>
                  </div>
                </div>
              )}

              {!isTriageRunning && lastTriageResult && (
                <div>
                  {activeAgentTab === 'perception' && (
                    <div className="flex flex-col gap-3">
                      <div className="text-cyan-400 border-b border-white/[0.06] pb-3 mb-2 font-sans font-semibold text-xs flex items-center justify-between">
                        <span>Perception Analysis Log</span>
                        <span className="text-zinc-500">Compiled</span>
                      </div>
                      <div className="whitespace-pre-wrap font-mono leading-relaxed">
                        {`Patient Record: ${lastTriageResult.name}\nTimestamp: ${lastTriageResult.timestamp}\n\nAcquired physiology:\n- Path: ${lastTriageResult.video_path}\n- Core metrics resolved:\n  • Heart Rate: ${metrics.bpm} BPM\n  • Respiration Rate: ${metrics.rr} breaths/min\n  • HRV: ${metrics.hrv.toFixed(1)} ms\n  • Stress Index: ${metrics.stress_index.toFixed(0)}\n  • Signal SNR: ${metrics.snr_db.toFixed(1)} dB\n\nDiagnostic buffer synchronized.`}
                      </div>
                    </div>
                  )}

                  {activeAgentTab === 'diagnostic' && (
                    <div className="flex flex-col gap-3">
                      <div className="text-amber-400 border-b border-white/[0.06] pb-3 mb-2 font-sans font-semibold text-xs flex items-center justify-between">
                        <span>Clinical Diagnostic Assessment</span>
                        <span className="text-zinc-500">Compiled</span>
                      </div>
                      <div className="whitespace-pre-wrap font-mono leading-relaxed">
                        {`Diagnosis pathology:\n- Recommended index: ESI Level ${lastTriageResult.esi_level}\n- Clinical focus: ${lastTriageResult.primary_diagnosis}\n- Compensated shock: ${lastTriageResult.is_shock ? "Shock criteria satisfied" : "Stable / no shock"}\n\nESI correlation logic:\n- Cross-correlation analysis: HR (${metrics.bpm}) × RR (${metrics.rr})\n- Clinical documentation resolved.\n\nRaw decision trace:\n${lastTriageResult.agent_output.substring(0, 1500)}...`}
                      </div>
                    </div>
                  )}

                  {activeAgentTab === 'coordinator' && (
                    <div className="flex flex-col gap-4 font-sans p-2">
                      <div className="text-indigo-400 border-b border-white/[0.06] pb-3 mb-2 font-sans font-semibold text-xs flex items-center justify-between">
                        <span>Dynamic Queue Allocation</span>
                        <span className="text-zinc-500">Compiled</span>
                      </div>

                      <div className="grid grid-cols-2 gap-4">
                        <div className="bg-white/[0.02] border border-white/[0.06] rounded-lg p-4 text-center">
                          <div className="text-[10px] text-zinc-500 font-semibold uppercase tracking-wide">Acuity Level</div>
                          <div className="text-3xl font-semibold text-indigo-400 mt-2 font-mono">Level {lastTriageResult.esi_level}</div>
                        </div>
                        <div className="bg-white/[0.02] border border-white/[0.06] rounded-lg p-4 text-center">
                          <div className="text-[10px] text-zinc-500 font-semibold uppercase tracking-wide">Priority Rating</div>
                          <div className="text-3xl font-semibold text-cyan-400 mt-2 font-mono">{lastTriageResult.priority_score}/100</div>
                        </div>
                      </div>

                      <div className="bg-white/[0.02] border border-white/[0.06] rounded-lg p-4">
                        <div className="text-[10px] text-zinc-500 font-semibold uppercase tracking-wide mb-2">Primary Clinical Indicator</div>
                        <div className="flex items-center gap-3">
                          <span className={`w-2.5 h-2.5 rounded-full ${lastTriageResult.is_shock ? 'bg-rose-500' : 'bg-emerald-500'}`}></span>
                          <span className={`text-sm font-medium ${lastTriageResult.is_shock ? 'text-rose-400 font-semibold' : 'text-zinc-200'}`}>
                            {lastTriageResult.primary_diagnosis} {lastTriageResult.is_shock && "(shock indicator active)"}
                          </span>
                        </div>
                      </div>

                      <div className="bg-white/[0.02] border border-white/[0.06] rounded-lg p-4">
                        <div className="text-[10px] text-zinc-500 font-semibold uppercase tracking-wide mb-2">Triage Summary &amp; Rationale</div>
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
        {activeTab === 'chat' && (() => {
          const selectedPatient = triageQueue.find(p => p.id === selectedPatientId) || null;
          return (
          <div className="bg-[#0d1117]/70 border border-white/[0.06] rounded-2xl p-6 backdrop-blur-xl shadow-lg flex flex-col h-[calc(100vh-48px)] relative">

            {/* ARIA Header */}
            <div className="flex items-center justify-between border-b border-white/[0.06] pb-5 mb-5 flex-shrink-0">
              <div className="flex items-center gap-4">
                <div className="w-11 h-11 rounded-xl bg-gradient-to-tr from-indigo-500 via-purple-500 to-pink-500 flex items-center justify-center shadow-md relative overflow-hidden shrink-0">
                  <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="white" strokeWidth="2"><path d="M12 2a10 10 0 110 20A10 10 0 0112 2z" opacity="0.3"/><path d="M12 6v6l4 2"/><circle cx="12" cy="12" r="2" fill="white"/></svg>
                </div>
                <div>
                  <div className="flex items-center gap-2">
                    <h2 className="text-base font-semibold text-white tracking-tight">ARIA</h2>
                    <span className="text-[9px] font-semibold uppercase tracking-wide bg-indigo-500/10 text-indigo-400 border border-indigo-500/20 px-2 py-0.5 rounded">Adaptive Real-time Intelligence for Acute-care</span>
                  </div>
                  <p className="text-[10px] text-zinc-500 font-medium uppercase tracking-wide mt-0.5">Powered by NVIDIA NIM · OCR document parsing · Multilingual · Evidence-based</p>
                </div>
              </div>
              <div className="flex items-center gap-2 text-[10px] font-semibold">
                <span className="w-1.5 h-1.5 rounded-full bg-indigo-400 animate-pulse"></span>
                <span className="text-indigo-400">ARIA Online</span>
              </div>
            </div>

            {/* Body: patient panel (left) + chat (right) */}
            <div className="flex-1 flex gap-5 min-h-0">

              {/* LEFT: patient record panel */}
              <div className="w-64 flex-shrink-0 flex flex-col gap-3 min-h-0">
                <div>
                  <label className="text-[10px] text-zinc-500 font-semibold uppercase tracking-wide">Patient record</label>
                  <select
                    value={selectedPatientId}
                    onChange={(e) => setSelectedPatientId(e.target.value)}
                    className="w-full mt-1.5 bg-white/[0.03] border border-white/[0.08] rounded-lg text-sm p-2.5 text-zinc-200 outline-none focus:border-cyan-500/50 transition-colors cursor-pointer"
                  >
                    {/* Native <option> popups render with a browser-controlled white
                        background on most platforms and ignore Tailwind classes on
                        the <select> for that popup — set explicit colors directly so
                        the list is readable instead of washed-out light-on-white. */}
                    <option value="" style={{ backgroundColor: '#111827', color: '#e5e7eb' }}>Select a patient...</option>
                    {triageQueue.map(p => (
                      <option key={p.id} value={p.id} style={{ backgroundColor: '#111827', color: '#e5e7eb' }}>{p.name || `Patient ${p.id.slice(0, 8)}`}</option>
                    ))}
                  </select>
                </div>

                <div className="flex-1 bg-white/[0.02] border border-white/[0.06] rounded-xl p-4 overflow-y-auto custom-scroll min-h-0">
                  {!selectedPatient ? (
                    <div className="h-full flex flex-col items-center justify-center text-center py-8">
                      <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.6" className="text-zinc-600 mb-2"><circle cx="12" cy="8" r="4"/><path d="M4 20c0-4 3.6-7 8-7s8 3 8 7"/></svg>
                      <p className="text-[11px] text-zinc-500 leading-relaxed">No patient selected. Choose one above, or mention a patient's name in your message and ARIA will select them automatically.</p>
                    </div>
                  ) : (
                    <div className="flex flex-col gap-3">
                      <div>
                        <p className="text-sm font-semibold text-white">{selectedPatient.name || `Patient ${selectedPatient.id.slice(0, 8)}`}</p>
                        <p className="text-[10px] text-zinc-500 font-mono mt-0.5">{selectedPatient.timestamp}</p>
                      </div>
                      <div className="flex items-center gap-2">
                        <span className={`text-[10px] font-semibold px-2 py-1 rounded border ${getEsiClass(selectedPatient.esi_level)}`}>
                          ESI {selectedPatient.esi_level}
                        </span>
                        <span className="text-[10px] font-semibold px-2 py-1 rounded bg-white/5 text-zinc-300 border border-white/10">
                          Score {selectedPatient.priority_score}
                        </span>
                      </div>
                      {selectedPatient.is_shock && (
                        <span className="text-[10px] font-semibold px-2 py-1 rounded bg-rose-500/10 text-rose-400 border border-rose-500/20 w-fit">
                          Compensated shock
                        </span>
                      )}

                      <div className="grid grid-cols-2 gap-2">
                        <div className="bg-white/[0.02] border border-white/[0.06] rounded-lg p-2.5 text-center">
                          <div className="text-[8.5px] text-zinc-500 font-semibold uppercase tracking-wide">Heart Rate</div>
                          <div className="text-sm font-semibold text-white font-mono mt-0.5">{selectedPatient.heart_rate != null ? Number(selectedPatient.heart_rate).toFixed(1) : '--'} <span className="text-[9px] text-zinc-500 font-medium">bpm</span></div>
                        </div>
                        <div className="bg-white/[0.02] border border-white/[0.06] rounded-lg p-2.5 text-center">
                          <div className="text-[8.5px] text-zinc-500 font-semibold uppercase tracking-wide">Respiration</div>
                          <div className="text-sm font-semibold text-white font-mono mt-0.5">{selectedPatient.respiration != null ? Number(selectedPatient.respiration).toFixed(1) : '--'} <span className="text-[9px] text-zinc-500 font-medium">br/min</span></div>
                        </div>
                        <div className="bg-white/[0.02] border border-white/[0.06] rounded-lg p-2.5 text-center">
                          <div className="text-[8.5px] text-zinc-500 font-semibold uppercase tracking-wide">HRV</div>
                          <div className="text-sm font-semibold text-white font-mono mt-0.5">{selectedPatient.hrv != null ? Number(selectedPatient.hrv).toFixed(1) : '--'} <span className="text-[9px] text-zinc-500 font-medium">ms</span></div>
                        </div>
                        <div className="bg-white/[0.02] border border-white/[0.06] rounded-lg p-2.5 text-center">
                          <div className="text-[8.5px] text-zinc-500 font-semibold uppercase tracking-wide">Stress Index</div>
                          <div className="text-sm font-semibold text-white font-mono mt-0.5">{selectedPatient.stress_index != null ? Number(selectedPatient.stress_index).toFixed(0) : '--'}</div>
                        </div>
                      </div>

                      <div>
                        <label className="text-[9px] text-zinc-500 font-semibold uppercase tracking-wide">Primary diagnosis</label>
                        <p className="text-xs text-zinc-300 mt-1">{selectedPatient.primary_diagnosis || 'N/A'}</p>
                      </div>
                      <div>
                        <label className="text-[9px] text-zinc-500 font-semibold uppercase tracking-wide">Triage summary</label>
                        <p className="text-xs text-zinc-400 mt-1 leading-relaxed">{selectedPatient.triage_summary || 'N/A'}</p>
                      </div>
                    </div>
                  )}
                </div>
              </div>

              {/* RIGHT: chat window */}
              <div className="flex-1 flex flex-col min-h-0">
                {/* Chat message bubbles */}
                <div className="flex-1 bg-white/[0.02] border border-white/[0.06] rounded-xl p-5 overflow-y-auto custom-scroll flex flex-col gap-4 mb-4 min-h-0">
                  {chatHistory.length === 0 ? (
                    <div className="h-full flex flex-col items-center justify-center text-zinc-550 text-center py-16">
                      <div className="w-14 h-14 rounded-2xl bg-gradient-to-tr from-indigo-500/15 to-purple-500/10 border border-indigo-500/20 flex items-center justify-center mb-4 mx-auto">
                        <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="rgb(129,140,248)" strokeWidth="1.5"><path d="M21 15a2 2 0 01-2 2H7l-4 4V5a2 2 0 012-2h14a2 2 0 012 2z"/></svg>
                      </div>
                      <p className="text-zinc-200 font-semibold text-sm">ARIA clinical intelligence ready</p>
                      <p className="text-[11px] text-zinc-500 mt-2 max-w-[460px] leading-relaxed font-medium">Ask clinical questions, request differential diagnoses, interpret vital signs, or attach photos of patient records and lab reports for instant OCR analysis.</p>
                      <div className="mt-5 flex flex-wrap gap-2 justify-center max-w-[500px]">
                        {['What does bradypnea indicate?', 'Explain ESI Level 2', 'Interpret high stress index', 'Normal rPPG heart rate range'].map(s => (
                          <button key={s} onClick={() => setChatInput(s)} className="text-[11px] bg-white/[0.03] hover:bg-white/[0.06] border border-white/10 text-indigo-300 px-3 py-1.5 rounded-lg font-medium transition-colors">{s}</button>
                        ))}
                      </div>
                    </div>
                  ) : (
                    chatHistory.map((msg, idx) => (
                      <div key={idx} className={`flex flex-col max-w-[85%] ${msg.role === 'user' ? 'self-end items-end' : 'self-start items-start'}`}>
                        <span className="text-[9px] text-zinc-500 font-semibold uppercase tracking-wide mb-1.5">
                          {msg.role === 'user' ? 'Clinician' : 'ARIA'}
                        </span>

                        <div className={`p-4 rounded-xl text-sm leading-relaxed font-medium whitespace-pre-wrap ${
                          msg.role === 'user'
                            ? 'bg-indigo-500/10 text-zinc-100 border border-indigo-500/20 rounded-tr-none'
                            : 'bg-white/[0.03] text-zinc-300 border border-white/[0.06] rounded-tl-none'
                        }`}>
                          {/* Attached Image inside Bubble */}
                          {msg.image && (
                            <div className="mb-3 max-w-[200px] rounded-lg overflow-hidden border border-white/10">
                              <img src={msg.image} alt="Clinical document" className="w-full h-auto object-cover" />
                            </div>
                          )}
                          {msg.content.replace(/^\[[a-z]{2}-[A-Z]{2}\]\s*/i, '').replace(/\*\*/g, '').replace(/\* /g, '• ')}
                        </div>
                      </div>
                    ))
                  )}

                  {isChatLoading && (
                    <div className="self-start flex items-center gap-2.5 text-xs text-cyan-400 font-medium font-mono">
                      <span className="w-2 h-2 rounded-full bg-cyan-400 animate-pulse"></span>
                      Decoding telemetry...
                    </div>
                  )}
                  <div ref={chatEndRef}></div>
                </div>

                {/* Input form */}
                <form onSubmit={handleSendChatMessage} className="flex flex-col gap-3 flex-shrink-0">
                  {/* Selected Image preview box */}
                  {selectedImage && (
                    <div className="flex items-center gap-3 p-3 bg-white/[0.03] border border-white/[0.08] rounded-xl">
                      <div className="w-12 h-12 rounded-lg overflow-hidden border border-white/10 relative">
                        <img src={selectedImage} alt="Attachment Preview" className="w-full h-full object-cover" />
                      </div>
                      <div className="flex-1 min-w-0">
                        <p className="text-xs font-semibold text-zinc-300 truncate">Clinical document attachment loaded</p>
                        <p className="text-[9px] text-zinc-500 font-mono uppercase tracking-wide font-medium">Ready for NVIDIA NIM OCR parsing</p>
                      </div>
                      <button
                        type="button"
                        onClick={() => setSelectedImage(null)}
                        className="text-rose-400 hover:text-rose-300 text-xs font-semibold p-2"
                      >
                        Remove
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
                      className="p-3 rounded-xl border border-white/[0.08] bg-white/[0.02] text-zinc-400 hover:text-zinc-200 hover:bg-white/[0.05] transition-colors"
                      title="Attach Document Photo"
                    >
                      <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M21.44 11.05l-9.19 9.19a6 6 0 01-8.49-8.49l9.19-9.19a4 4 0 015.66 5.66l-9.2 9.19a2 2 0 01-2.83-2.83l8.49-8.48"/></svg>
                    </button>
                    <button
                      type="button"
                      onClick={toggleRecording}
                      className={`p-3 rounded-xl border transition-colors ${
                        isRecording ? 'bg-rose-500/10 text-rose-400 border-rose-500/30 animate-pulse' : 'bg-white/[0.02] text-zinc-400 border-white/[0.08] hover:bg-white/[0.05]'
                      }`}
                      title="Voice Input (STT)"
                    >
                      <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M12 1a3 3 0 00-3 3v8a3 3 0 006 0V4a3 3 0 00-3-3z"/><path d="M19 10v2a7 7 0 01-14 0v-2M12 19v4"/></svg>
                    </button>
                    <input
                      type="text"
                      value={chatInput}
                      onChange={(e) => setChatInput(e.target.value)}
                      placeholder={selectedImage ? "Add clinical context or ask ARIA to parse this document..." : "Ask ARIA about vital signs, diagnoses, ESI levels, or clinical protocols..."}
                      className="flex-1 bg-white/[0.02] border border-white/[0.08] rounded-xl text-sm px-4 text-zinc-200 outline-none focus:border-cyan-500/50 transition-colors"
                    />
                    <button
                      type="submit"
                      disabled={isChatLoading || (!chatInput.trim() && !selectedImage)}
                      className="bg-indigo-500/10 hover:bg-indigo-500/20 text-indigo-400 border border-indigo-500/25 font-semibold text-xs py-3 px-6 rounded-xl transition-colors disabled:opacity-40 disabled:cursor-not-allowed"
                    >
                      Send
                    </button>
                    <button
                      type="button"
                      onClick={() => {
                        if (ttsEnabled) window.speechSynthesis?.cancel();
                        setTtsEnabled(v => !v);
                      }}
                      title={ttsEnabled ? 'Voice replies on — click to mute' : 'Voice replies muted — click to unmute'}
                      className={`p-3 rounded-xl border transition-colors ${
                        ttsEnabled ? 'bg-cyan-500/10 text-cyan-400 border-cyan-500/30' : 'bg-white/[0.02] text-zinc-500 border-white/[0.08] hover:bg-white/[0.05]'
                      }`}
                    >
                      {ttsEnabled ? (
                        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><polygon points="11 5 6 9 2 9 2 15 6 15 11 19 11 5"/><path d="M15.54 8.46a5 5 0 010 7.07M19.07 4.93a10 10 0 010 14.14"/></svg>
                      ) : (
                        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><polygon points="11 5 6 9 2 9 2 15 6 15 11 19 11 5"/><line x1="23" y1="9" x2="17" y2="15"/><line x1="17" y1="9" x2="23" y2="15"/></svg>
                      )}
                    </button>
                  </div>
                </form>
              </div>
            </div>
          </div>
          );
        })()}

      </main>
      </div>

    </div>
  );
}
