using System;
using System.Collections.Generic;
using System.IO;
using UnityEngine;

namespace Resources.Scripts
{
    /// <summary>
    /// Logs gaze focus and pinch-selection events to a CSV file.
    /// Records whether gaze came from Tobii eye tracking or head-gaze fallback.
    /// Attach to any scene object and wire up events from GazeRaycaster and PinchDetector.
    /// </summary>
    public class ExperimentLogger : MonoBehaviour
    {
        [Header("Configuration")]
        [Tooltip("Participant / session identifier.")]
        public string participantId = "P001";

        [Tooltip("Output directory for log files. Relative to Application.persistentDataPath.")]
        public string outputFolder = "ExperimentLogs";

        [Header("References")]
        public GazeRaycaster gazeRaycaster;

        private string _filePath;
        private float _sessionStartTime;

        private readonly List<LogEntry> _entries = new List<LogEntry>();

        public struct LogEntry
        {
            public float Timestamp;
            public string EventType;     // "GAZE_ENTER", "GAZE_EXIT", "SELECT"
            public string StimulusId;
            public string GazeSource;    // "EyeTracking", "HeadGaze", "None"
            public Vector3 GazeOrigin;
            public Vector3 GazeDirection;
            public Vector3 HeadPosition;
            public Vector3 HeadForward;
        }

        private void Start()
        {
            _sessionStartTime = Time.time;

            string dir = Path.Combine(Application.persistentDataPath, outputFolder);
            Directory.CreateDirectory(dir);

            string timestamp = DateTime.Now.ToString("yyyyMMdd_HHmmss");
            _filePath = Path.Combine(dir, $"log_{participantId}_{timestamp}.csv");

            // CSV header
            string header = string.Join(",",
                "Timestamp", "ElapsedSec", "EventType", "StimulusId", "GazeSource",
                "GazeOriginX", "GazeOriginY", "GazeOriginZ",
                "GazeDirX", "GazeDirY", "GazeDirZ",
                "HeadPosX", "HeadPosY", "HeadPosZ",
                "HeadFwdX", "HeadFwdY", "HeadFwdZ"
            );
            File.WriteAllText(_filePath, header + "\n");

            Debug.Log($"[ExperimentLogger] Logging to: {_filePath}");

            // Auto-subscribe to gaze events
            if (gazeRaycaster != null)
            {
                gazeRaycaster.onStimulusFocused.AddListener(OnGazeEnter);
                gazeRaycaster.onStimulusUnfocused.AddListener(OnGazeExit);
            }
        }

        /// <summary>Call from PinchDetector.onStimulusClicked.</summary>
        public void OnStimulusSelected(Interactable stimulus)
        {
            RecordEvent("SELECT", stimulus);
        }

        public void OnGazeEnter(Interactable stimulus)
        {
            RecordEvent("GAZE_ENTER", stimulus);
        }

        public void OnGazeExit(Interactable stimulus)
        {
            RecordEvent("GAZE_EXIT", stimulus);
        }

        private void RecordEvent(string eventType, Interactable stimulus)
        {
            float elapsed = Time.time - _sessionStartTime;
            string stimId = stimulus != null ? stimulus.StimulusId : "NONE";

            // Gaze data from the raycaster
            Vector3 gazeOrigin = Vector3.zero;
            Vector3 gazeDir = Vector3.forward;
            string gazeSource = "None";

            if (gazeRaycaster != null)
            {
                gazeOrigin = gazeRaycaster.CurrentGazeOrigin;
                gazeDir = gazeRaycaster.CurrentGazeDirection;
                gazeSource = gazeRaycaster.ActiveSource.ToString();
            }

            // Head pose (always from the camera transform)
            Transform cam = gazeRaycaster != null ? gazeRaycaster.transform : Camera.main.transform;
            Vector3 headPos = cam.position;
            Vector3 headFwd = cam.forward;

            var entry = new LogEntry
            {
                Timestamp = elapsed,
                EventType = eventType,
                StimulusId = stimId,
                GazeSource = gazeSource,
                GazeOrigin = gazeOrigin,
                GazeDirection = gazeDir,
                HeadPosition = headPos,
                HeadForward = headFwd
            };
            _entries.Add(entry);

            string line = string.Format(
                "{0:F4},{1:F4},{2},{3},{4},{5:F4},{6:F4},{7:F4},{8:F4},{9:F4},{10:F4},{11:F4},{12:F4},{13:F4},{14:F4},{15:F4},{16:F4}",
                Time.time, elapsed, eventType, stimId, gazeSource,
                gazeOrigin.x, gazeOrigin.y, gazeOrigin.z,
                gazeDir.x, gazeDir.y, gazeDir.z,
                headPos.x, headPos.y, headPos.z,
                headFwd.x, headFwd.y, headFwd.z
            );

            try
            {
                File.AppendAllText(_filePath, line + "\n");
            }
            catch (Exception e)
            {
                Debug.LogError($"[ExperimentLogger] Write error: {e.Message}");
            }
        }

        private void OnDestroy()
        {
            Debug.Log($"[ExperimentLogger] Session ended. {_entries.Count} events recorded to {_filePath}");
        }
    }
}