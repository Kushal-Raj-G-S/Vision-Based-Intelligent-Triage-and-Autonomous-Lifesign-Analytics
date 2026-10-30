'use client';

import React, { useState, useEffect, useRef } from 'react';
import { MonitorBar, ParamBlock, SweepTrace, deriveAlarms, LIMITS, PARAM } from '@/components/BedsideMonitor';
import AppRail from '@/components/AppRail';

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
  has_source?: boolean;
  calibration_done: boolean;
  ppg_signal: number[];
  calibration_progress: number;
  face_detected?: boolean;
  video_current_sec?: number | null;
  video_total_sec?: number | null;
}

const BACKEND_URL = "http://127.0.0.1:5002";

// Live acquisition window before auto-triage fires. Was 30s — too short for
// respiratory rate, which needs a 60s buffer just to attempt its first
// reading (core/rppg.py's _RR_WINDOW_SEC) plus more time beyond that to
// accumulate the 3+ readings the validated session-median accuracy actually
// depends on (measured at 1.59-2.51 MAE on real ground truth, vs. no signal
// at all before 60s). 90s gives ~30s of margin past RR's hard floor; HR/HRV/
// Stress Index/SQI are all ready well within that window regardless.
const SESSION_WINDOW_SEC = 90;

type NavigationTab = 'monitor' | 'queue' | 'crew' | 'chat';

export default function Home() {
  // Navigation
  const [activeTab, setActiveTab] = useState<NavigationTab>('monitor');
  // Colour-magnified pulse overlay on the live feed (display only; see core/pulse_viz.py)
  const [pulseView, setPulseView] = useState(false); // experimental: off by default
  const [detailsOpen, setDetailsOpen] = useState(false);

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
  // Live session timer and auto-report — see SESSION_WINDOW_SEC for why 90s
  const [sessionTimer, setSessionTimer] = useState<number>(SESSION_WINDOW_SEC);
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

  // Countdown timer (SESSION_WINDOW_SEC) — starts when calibration_done flips true
  useEffect(() => {
    if (metrics.calibration_done && metrics.is_live) {
      // Start countdown if not already running
      if (timerRef.current === null && sessionTimer <= SESSION_WINDOW_SEC && !reportFetchedRef.current) {
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
                // Release the webcam hardware the moment the 90s window ends
                // — the buffered signal is already enough to generate the
                // report, there's no reason to keep the camera lit past the
                // acquisition window while triage runs.
                handleReleaseCamera();
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
        setSessionTimer(SESSION_WINDOW_SEC);
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
  // Reset the 90s countdown at the moment the user actually presses play,
  // rather than leaving it to the calibration_done effect below — that
  // effect only resets the timer while calibration_done is still false,
  // so a leftover low count from a just-ended prior session could survive
  // into the next one's first few polls and start counting down from
  // wherever it left off instead of a fresh SESSION_WINDOW_SEC.
  const resetSessionWindow = () => {
    if (timerRef.current) { clearInterval(timerRef.current); timerRef.current = null; }
    setSessionTimer(SESSION_WINDOW_SEC);
    setSessionReport(null);
    reportFetchedRef.current = false;
    setTriageAutoRan(false);
  };

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
        resetSessionWindow();
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
    <div className="vital-app flex-1 font-sans min-h-screen antialiased transition-colors duration-200">
      
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
        .shadow-glow-red {
          box-shadow: 0 1px 3px rgba(0, 0, 0, 0.4);
        }
      `}</style>

      {/* 1 &amp; 2. Sidebar Navigation (collapsible) */}
      <div className="flex">
        <AppRail active={activeTab} onSelect={goToTab} queueCount={triageQueue.length} backendOnline={backendOnline} />

      {/* 3. Main Workspace Container */}
      <main className={`flex-1 min-w-0 max-w-[1700px] mx-auto px-6 mt-4 pb-4`}>

        <header className={`flex flex-wrap items-end justify-between gap-4 hidden`}>
          <div>
            <h1 className="text-2xl font-semibold tracking-tight text-white">
              {activeTab === 'monitor' ? 'Vitals monitor' : activeTab === 'queue' ? 'Triage dispatch' : activeTab === 'crew' ? 'Clinical crew' : 'ARIA assistant'}
            </h1>
            <p className={`text-sm text-zinc-500 mt-1 ${activeTab === 'monitor' ? 'hidden' : ''}`}>
              {activeTab === 'monitor' ? 'Live contactless heart rate, respiration and stress from the camera.' :
               activeTab === 'queue' ? 'Everyone assessed today, ordered by acuity.' :
               activeTab === 'crew' ? 'How the three agents reach and explain a triage decision.' :
               'Ask ARIA about the current patient or the queue.'}
            </p>
          </div>
          <div className="flex items-center gap-2 text-xs font-medium">
            <span className={`inline-flex items-center gap-1.5 px-3 py-1.5 rounded-full border ${backendOnline ? 'border-emerald-500/30 bg-emerald-500/10 text-emerald-400' : 'border-rose-500/30 bg-rose-500/10 text-rose-400'}`}>
              <span className={`w-1.5 h-1.5 rounded-full ${backendOnline ? 'bg-emerald-400' : 'bg-rose-500'}`}></span>
              {backendOnline ? 'Backend connected' : 'Backend offline'}
            </span>
            <span className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-full border border-white/10 bg-white/[0.04] text-zinc-300">
              {triageQueue.length} in queue
            </span>
          </div>
        </header>


        {/* VIEW 1: LIVE VITAL MONITOR — optical bedside monitor, one screen, no page scroll */}
        {activeTab === 'monitor' && (() => {
          const alarms = deriveAlarms({
            bpm: metrics.bpm, hrClass: metrics.classification, rr: metrics.rr, rrClass: metrics.rr_classification,
            stress: metrics.stress_index, warnings: metrics.warnings || [], faceDetected: metrics.face_detected, hasSource: metrics.has_source,
          });
          const hrC = (metrics.classification || '').toUpperCase();
          const rrC = (metrics.rr_classification || '').toUpperCase();
          return (
          <div className="bm-root">
            <MonitorBar
              patient={patientName.trim()}
              source={!metrics.has_source ? 'No source' : deviceMode === 'upload' ? 'Recorded video' : 'Live camera'}
              live={!!metrics.is_live}
              alarms={alarms}
              scanLeft={metrics.calibration_done && sessionTimer > 0 ? sessionTimer : null}
              scanTotal={SESSION_WINDOW_SEC}
              onDetails={() => setDetailsOpen(true)}
              detailsBadge={(metrics.warnings || []).length + (sessionReport ? 1 : 0)}
            />
            <div className="bm-grid">
              <div className="bm-left">
                <div className="bm-view">
                  <div className="bm-view-media group">
                    {backendOnline ? (
                      metrics.status === 'IMAGE_READY' || metrics.remark === 'IMAGE_DEMO' ? (
                        <img src={`${BACKEND_URL}/image_feed`} alt="Intake snapshot" />
                      ) : (
                        <img
                          key={`${videoFeedKey}-${pulseView}`}
                          src={`${BACKEND_URL}/video_feed?t=${videoFeedKey}${pulseView ? '&pulse=1' : ''}`}
                          alt="Patient camera"
                          onError={() => { setTimeout(() => setVideoFeedKey(k => k + 1), 1000); }}
                        />
                      )
                    ) : (
                      <div className="bm-empty"><strong>Backend offline</strong><span>Start the API server on port 5002 to see the camera.</span></div>
                    )}
                    {backendOnline && metrics.has_source === false && (
                      <div className="bm-empty">
                        <strong>No patient on monitor</strong>
                        <span>Enter a name, then press Start to use the camera or Load video for a recording.</span>
                      </div>
                    )}
                  </div>

                  <div className="bm-view-label">
                    <span className={metrics.status === 'OK' ? 'ok' : ''}>{({
                      OK: 'Signal OK',
                      CALIBRATING: `Calibrating ${metrics.calibration_progress || 0}%`,
                      OUT_OF_RANGE: 'Reading outside 48-150 bpm',
                      LOW_SIGNAL: 'Weak signal',
                      NO_FACE: 'No face',
                      WAITING: 'Waiting',
                      VIDEO_ENDED: 'Video ended',
                      DISCONNECTED: 'Disconnected',
                      IMAGE_READY: 'Still image',
                    } as Record<string, string>)[metrics.status] ?? metrics.status}</span>
                    {metrics.face_detected && <span className="ok">Face locked</span>}
                    {metrics.is_live && <span className="rec">Recording</span>}
                    {metrics.video_total_sec != null && <span>{formatDuration(metrics.video_current_sec || 0)} / {formatDuration(metrics.video_total_sec)}</span>}
                  </div>

                  {(metrics.status === 'VIDEO_ENDED' && !metrics.is_live) && (
                    <div className="bm-toast">
                      <span style={{ color: '#ffcf4a' }}>{isTriageRunning ? 'Video finished. Triage crew is running…' : 'Video finished. Triage crew ran on the full video.'}</span>
                      <button className="bm-btn" onClick={() => goToTab('crew')}>Open crew</button>
                    </div>
                  )}
                  {uploadMessage.text && (
                    <div className="bm-toast" style={{ top: 100, color: uploadMessage.type === 'success' ? '#4ef0a0' : '#ff8a96' }}>{uploadMessage.text}</div>
                  )}

                  <div className="bm-dock">
                    <label htmlFor="bm-patient" className="sr-only">Patient full name</label>
                    <input id="bm-patient" className="bm-field" type="text" placeholder="Patient full name" value={patientName} onChange={(e) => setPatientName(e.target.value)} />
                    <label htmlFor="bm-source" className="sr-only">Camera or recorded video</label>
                    <select
                      id="bm-source" className="bm-field"
                      value={deviceMode === 'upload' ? '__upload__' : (selectedCamera !== null ? selectedCamera : '')}
                      onChange={(e) => { if (e.target.value === '__upload__') { setDeviceMode('upload'); } else { setDeviceMode('camera'); setSelectedCamera(Number(e.target.value)); } }}
                    >
                      {cameras.map((cam) => (<option key={cam.index} value={cam.index}>{cam.label}</option>))}
                      {cameras.length === 0 && <option value="">No camera found</option>}
                      <option value="__upload__">Recorded video</option>
                    </select>
                    {deviceMode === 'upload' && (
                      <>
                        <label htmlFor="bm-video" className="sr-only">Which recording</label>
                        <select id="bm-video" className="bm-field" value={selectedUploadedVideo} onChange={(e) => setSelectedUploadedVideo(e.target.value)}>
                          {uploadedVideos.length === 0 && <option value="">No videos uploaded</option>}
                          {uploadedVideos.map((v) => (<option key={v.filename} value={v.filename}>{v.filename}</option>))}
                        </select>
                      </>
                    )}
                    {(() => {
                      const running = !!metrics.is_live || (!!metrics.has_source && metrics.status !== 'VIDEO_ENDED');
                      return (
                        <button
                          className={`bm-btn ${running ? 'stop' : 'go'}`}
                          onClick={running ? handleReleaseCamera : (deviceMode === 'upload' ? handleStartUploadedVideo : handleStartWebcam)}
                        >{running ? 'Stop' : (deviceMode === 'upload' ? 'Load video' : 'Start')}</button>
                      );
                    })()}
                    <label className="bm-btn" style={{ display: 'inline-flex', alignItems: 'center' }}>
                      Upload
                      <input type="file" accept="video/*,image/*" onChange={handleFileUpload} className="absolute inset-0 opacity-0 cursor-pointer w-full h-full" aria-label="Upload a video or image" />
                    </label>
                    <button className={`bm-btn ${pulseView ? 'on' : ''}`} onClick={() => setPulseView(v => !v)} title="Experimental: colour-magnify the pulse in the face">
                      Pulse view{pulseView ? ' on' : ''}
                    </button>
                  </div>
                </div>

                <SweepTrace
                  samples={metrics.ppg_signal || []}
                  color={PARAM.hr}
                  label="Pleth · camera rPPG"
                  rate={metrics.sqi > 0 ? `SQI ${Number(metrics.sqi).toFixed(0)}%` : 'calibrating'}
                />
              </div>

              <div className="bm-right">
                <ParamBlock
                  tag="HR" unit="bpm" color={PARAM.hr}
                  value={metrics.bpm > 0 ? metrics.bpm.toFixed(0) : '–?–'} hasValue={metrics.bpm > 0}
                  hi={LIMITS.HR_HI} lo={LIMITS.HR_LO}
                  status={metrics.bpm > 0 ? (hrC === 'NORMAL' ? 'Normal' : hrC === 'TACHYCARDIA' ? 'Tachycardia' : hrC === 'BRADYCARDIA' ? 'Bradycardia' : hrC.toLowerCase()) : (metrics.calibration_done ? 'Searching' : 'Calibrating')}
                  conf={metrics.bpm > 0 ? Number(metrics.confidence) || 0 : null}
                  alarm={metrics.bpm > 0 && (hrC === 'TACHYCARDIA' || hrC === 'BRADYCARDIA')}
                  pulseSec={metrics.bpm > 0 ? 60 / metrics.bpm : undefined}
                  trend={metrics.bpm}
                />
                <ParamBlock
                  tag="RR" unit="breaths/min" color={PARAM.rr}
                  value={metrics.rr > 0 ? Number(metrics.rr).toFixed(0) : '–?–'} hasValue={metrics.rr > 0}
                  hi={LIMITS.RR_HI} lo={12}
                  status={metrics.rr > 0 ? (rrC === 'NORMAL' ? 'Normal' : rrC === 'TACHYPNEA' ? 'Tachypnoea' : rrC === 'BRADYPNEA' ? 'Low rate' : rrC.toLowerCase()) : 'Needs ~30 s of signal'}
                  conf={metrics.rr > 0 ? Number(metrics.rr_confidence) || 0 : null}
                  foot={metrics.rr > 0 ? `${Number(metrics.rr).toFixed(1)}` : undefined}
                  breatheSec={metrics.rr > 0 ? 60 / metrics.rr : undefined}
                  trend={Number(metrics.rr) || 0}
                />
                <div className="bm-pair">
                  <ParamBlock
                    tag="STRESS" unit="Baevsky SI" color={PARAM.stress} size="lg"
                    value={metrics.stress_index > 0 ? Number(metrics.stress_index).toFixed(0) : '–?–'} hasValue={metrics.stress_index > 0}
                    hi={LIMITS.STRESS_HI} lo={null}
                    status={metrics.stress_index > LIMITS.STRESS_HI ? 'Critical' : metrics.stress_index > LIMITS.STRESS_ELEV ? 'Elevated' : metrics.stress_index > 0 ? 'Optimal' : 'Waiting'}
                    foot={Number(metrics.hrv) > 0 ? `HRV ${Number(metrics.hrv).toFixed(0)} ms` : undefined}
                  />
                  <ParamBlock
                    tag="SQI" unit="signal quality %" color={PARAM.sqi} size="lg"
                    value={metrics.sqi > 0 ? Number(metrics.sqi).toFixed(0) : '–?–'} hasValue={metrics.sqi > 0}
                    hi={null} lo={LIMITS.SQI_MIN}
                    status={metrics.sqi >= LIMITS.SQI_MIN ? 'Usable' : metrics.sqi > 0 ? 'Low confidence' : 'Waiting'}
                    foot={metrics.snr_db ? `SNR ${Number(metrics.snr_db).toFixed(1)} dB` : undefined}
                  />
                </div>
              </div>
            </div>

            {/* ── DETAILS DRAWER: telemetry, warnings, 30-second report ── */}
            {detailsOpen && (
              <div className="fixed inset-0 z-[60] flex justify-end" onClick={() => setDetailsOpen(false)}>
                <div className="absolute inset-0 bg-black/45 backdrop-blur-[2px]" />
                <aside onClick={(e) => e.stopPropagation()} className="relative h-full w-full max-w-[460px] overflow-y-auto border-l border-white/10 p-5 shadow-2xl custom-scroll" style={{ background: 'var(--bg)' }}>
                  <div className="mb-4 flex items-center justify-between">
                    <h2 className="text-base font-semibold text-white">Session details</h2>
                    <button onClick={() => setDetailsOpen(false)} className="rounded-md border border-white/10 px-2.5 py-1 text-sm text-zinc-400 hover:text-white">✕</button>
                  </div>
                  <div className="mb-4 space-y-2 rounded-xl border border-white/[0.07] bg-white/[0.02] p-4 font-mono text-xs">
                    <div className="flex justify-between"><span className="text-zinc-500">Camera</span><span className={metrics.is_live ? 'text-emerald-400' : 'text-zinc-500'}>{metrics.is_live ? 'ACTIVE' : 'OFFLINE'}</span></div>
                    <div className="flex justify-between"><span className="text-zinc-500">Face ROI</span><span className={metrics.face_detected ? 'text-cyan-400' : 'text-zinc-500'}>{metrics.face_detected ? 'LOCKED' : 'NO LOCK'}</span></div>
                    <div className="flex justify-between"><span className="text-zinc-500">Ambient light</span><span className="text-white">{metrics.estimated_lux} lux ({metrics.estimated_lux > 100 ? 'optimal' : 'low'})</span></div>
                    <div className="flex justify-between"><span className="text-zinc-500">Stability</span><span className="text-white">{metrics.stability_indicator} ({Number(metrics.stability).toFixed(1)} bpm)</span></div>
                    <div className="flex justify-between"><span className="text-zinc-500">SNR</span><span className="text-white">{metrics.snr_db > 0 ? `${Number(metrics.snr_db).toFixed(1)} dB` : '--'}</span></div>
                  </div>
                  {metrics.warnings && metrics.warnings.length > 0 && (
                    <div className="mb-4 rounded-xl border border-rose-500/25 bg-rose-500/[0.07] p-4">
                      <div className="mb-1.5 text-xs font-semibold text-rose-400">⚠ Acquisition warnings</div>
                      <ul className="list-inside list-disc text-xs font-medium leading-relaxed text-rose-300/90">{metrics.warnings.map((w, idx) => <li key={idx}>{w}</li>)}</ul>
                    </div>
                  )}
                  {isTriageRunning && <div className="mb-4 rounded-xl border border-indigo-500/30 bg-indigo-500/10 p-4 text-xs font-semibold text-indigo-300 animate-pulse">Clinical Crew running: diagnostics, ESI acuity check and saving the record…</div>}
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
                  {!sessionReport && <p className="text-xs text-zinc-500">The 30-second clinical report appears here after a full scan window.</p>}
                </aside>
              </div>
            )}
          </div>
          );
        })()}

        {/* VIEW 2: TRIAGE DISPATCH — acuity-sorted list + selected patient, one screen */}
        {activeTab === 'queue' && (() => {
          const sel = triageQueue.find(p => p.id === expandedPatientId) || triageQueue[0] || null;
          const counts = [1, 2, 3, 4, 5].map(l => triageQueue.filter(p => p.esi_level === l).length);
          const total = triageQueue.length || 1;
          const esiName = ['', 'Resuscitation', 'Emergent', 'Urgent', 'Less urgent', 'Non-urgent'];
          return (
          <div className="tq-root">
            <div className="tq-head">
              <div>
                <h1 className="tq-title">Triage dispatch</h1>
                <p className="tq-sub">{triageQueue.length} assessed · sorted by acuity, ESI 1 first</p>
              </div>
              <div className="tq-dist" aria-label="Patients by ESI level">
                {counts.map((n, i) => n > 0 && (
                  <span key={i} className={`tq-dist-seg esi-${i + 1}`} style={{ flexGrow: n / total }} title={`ESI ${i + 1}: ${n}`}>
                    <b>{n}</b> ESI {i + 1}
                  </span>
                ))}
              </div>
              <button onClick={handleClearQueue} className="tq-flush">Clear queue</button>
            </div>

            {triageQueue.length === 0 ? (
              <div className="tq-empty">
                <strong>No one is waiting</strong>
                <span>Each completed scan lands here with its ESI level, ordered by acuity.</span>
              </div>
            ) : (
            <div className="tq-grid">
              <div className="tq-list custom-scroll" role="listbox" aria-label="Patients">
                {triageQueue.map((p) => (
                  <button
                    key={p.id} role="option" aria-selected={sel?.id === p.id}
                    onClick={() => setExpandedPatientId(p.id)}
                    className={`tq-row ${sel?.id === p.id ? 'on' : ''}`}
                  >
                    <span className={`tq-orb esi-${p.esi_level}`}>{p.esi_level}</span>
                    <span className="tq-row-main">
                      <span className="tq-row-name">{p.name}{p.is_shock && <i className="tq-shock">Shock</i>}</span>
                      <span className="tq-row-dx">{p.primary_diagnosis}</span>
                    </span>
                    <span className="tq-row-vitals">
                      <span style={{ color: 'var(--sig-hr)' }}>{p.heart_rate != null ? Number(p.heart_rate).toFixed(0) : '–'}</span>
                      <span style={{ color: 'var(--sig-rr)' }}>{p.respiration != null ? Number(p.respiration).toFixed(0) : '–'}</span>
                    </span>
                    <span className="tq-row-time">{(p.timestamp || '').slice(11, 16)}</span>
                  </button>
                ))}
              </div>

              {sel && (
                <article className="tq-detail custom-scroll">
                  <div className="tq-detail-top">
                    <span className={`tq-orb big esi-${sel.esi_level}`}>{sel.esi_level}</span>
                    <div className="min-w-0">
                      <div className="tq-detail-esi">ESI {sel.esi_level} · {esiName[sel.esi_level] || ''}</div>
                      <h2 className="tq-detail-name">{sel.name}</h2>
                      <div className="tq-detail-meta">{sel.timestamp} · priority {sel.priority_score}/100</div>
                    </div>
                  </div>
                  {sel.is_shock && <div className="tq-alert">Compensated shock pattern: manual review required</div>}
                  <div className="tq-dx">{sel.primary_diagnosis}</div>
                  <div className="tq-vitals">
                    {[
                      { k: 'HR', u: 'bpm', v: sel.heart_rate, d: 0, c: 'var(--sig-hr)' },
                      { k: 'RR', u: 'br/min', v: sel.respiration, d: 1, c: 'var(--sig-rr)' },
                      { k: 'HRV', u: 'ms', v: sel.hrv, d: 0, c: 'var(--sig-sqi)' },
                      { k: 'Stress', u: 'index', v: sel.stress_index, d: 0, c: 'var(--sig-stress)' },
                    ].map(x => (
                      <div key={x.k} className="tq-vital" style={{ ['--pc' as string]: x.c }}>
                        <span className="tq-vital-k">{x.k}</span>
                        <span className="tq-vital-v">{x.v != null ? Number(x.v).toFixed(x.d) : '–'}</span>
                        <span className="tq-vital-u">{x.u}</span>
                      </div>
                    ))}
                  </div>
                  <p className="tq-summary">{sel.triage_summary}</p>
                  <div className="tq-foot">Record {sel.id} · {sel.video_path}</div>
                </article>
              )}
            </div>
            )}
          </div>
          );
        })()}

        {/* VIEW 3: CLINICAL CREW — the pipeline as it really runs: the engine decides, agents explain */}
        {activeTab === 'crew' && (
          <div className="cr-root">
            <div className="tq-head">
              <div>
                <h1 className="tq-title">Clinical crew</h1>
                <p className="tq-sub">The deterministic engine sets ESI, shock and priority. Three agents explain it.</p>
              </div>
              <button
                onClick={() => handleRunTriage()}
                disabled={isTriageRunning || metrics.status === 'DISCONNECTED' || triageAutoRan}
                title={triageAutoRan ? 'Triage already ran automatically for this captured session' : undefined}
                className="cr-run"
              >
                {isTriageRunning ? 'Running…' : triageAutoRan ? 'Already dispatched' : 'Run triage crew'}
              </button>
            </div>

            <div className={`cr-flow ${isTriageRunning ? 'live' : ''}`}>
              <div className="cr-node src">
                <span className="cr-node-k">Camera vitals</span>
                <span className="cr-node-v">{lastTriageResult?.heart_rate != null ? `${Number(lastTriageResult.heart_rate).toFixed(0)} bpm` : metrics.bpm > 0 ? `${metrics.bpm.toFixed(0)} bpm` : '—'}</span>
                <span className="cr-node-s">rPPG · HR, RR, HRV, stress</span>
              </div>
              <span className="cr-link" aria-hidden />
              <div className="cr-node engine">
                <span className="cr-node-k">Clinical engine</span>
                <span className="cr-node-v">{lastTriageResult ? `ESI ${lastTriageResult.esi_level} · ${lastTriageResult.priority_score}` : '—'}</span>
                <span className="cr-node-s">Rules, no AI · decides acuity</span>
              </div>
              <span className="cr-link" aria-hidden />
              {([
                ['perception', 'Perception agent', 'Reads the signals'],
                ['diagnostic', 'Diagnostic agent', 'Explains the finding'],
                ['coordinator', 'Coordinator agent', 'Writes the handover'],
              ] as const).map(([id, k, s], i) => (
                <React.Fragment key={id}>
                  {i > 0 && <span className="cr-link" aria-hidden />}
                  <button className={`cr-node agent ${activeAgentTab === id ? 'on' : ''}`} onClick={() => setActiveAgentTab(id)} aria-pressed={activeAgentTab === id}>
                    <span className="cr-node-k">{k}</span>
                    <span className="cr-node-v">{lastTriageResult ? 'Done' : isTriageRunning ? 'Working' : 'Idle'}</span>
                    <span className="cr-node-s">{s}</span>
                  </button>
                </React.Fragment>
              ))}
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
          <div className="ch-root">

            {/* ARIA Header */}
            <div className="tq-head">
              <div className="flex items-center gap-4">
                <span className="ch-avatar" aria-hidden>
                  <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M4 5h16v11H9l-5 4z" /><path d="M8 10h.01M12 10h.01M16 10h.01" /></svg>
                </span>
                <div>
                  <h1 className="tq-title">ARIA assistant</h1>
                  <p className="tq-sub">Ask about a patient, a vital sign or a protocol. ARIA explains; the clinical engine decides acuity.</p>
                </div>
              </div>
              <span className="ch-online"><i />Online · NVIDIA NIM</span>
            </div>

            {/* Body: patient panel (left) + chat (right) */}
            <div className="ch-body">

              {/* LEFT: patient record panel */}
              <div className="ch-side">
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

                <div className="ch-card custom-scroll">
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
              <div className="ch-main">
                {/* Chat message bubbles */}
                <div className="ch-thread custom-scroll">
                  {chatHistory.length === 0 ? (
                    <div className="h-full flex flex-col items-center justify-center text-zinc-550 text-center py-16">
                      <div className="ch-avatar big mb-4 mx-auto">
                        <svg width="26" height="26" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8"><path d="M21 15a2 2 0 01-2 2H7l-4 4V5a2 2 0 012-2h14a2 2 0 012 2z"/></svg>
                      </div>
                      <p className="ch-empty-title">What would you like to know?</p>
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
                        <span className="ch-who">{msg.role === 'user' ? 'You' : 'ARIA'}</span>

                        <div className={`ch-msg ${msg.role === 'user' ? 'me' : 'ai'}`}>
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

                  <div className="ch-composer">
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
                      className="ch-input"
                    />
                    <button
                      type="submit"
                      disabled={isChatLoading || (!chatInput.trim() && !selectedImage)}
                      className="cr-run ch-send"
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
