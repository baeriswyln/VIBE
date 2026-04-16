using System;
using System.Collections.Generic;
using System.IO;
using LSL;
using UnityEngine;

namespace Resources.Scripts.Logging
{
    /// <summary>
    /// Logs all events that are sent to the logger - writes to a CSV file, and publishes via the LSL.
    /// Attach to any scene object and wire up events.
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
        public PinchDetector pinchDetector;
        
        [Header("LSL Configuration")]
        string StreamName = "LSL4Unity.TechDemo.EventStream";
        string StreamType = "Markers";
        private StreamOutlet outlet;
        private string[] sample = { "" };
        
        private string _filePath;
        private float _sessionStartTime;

        private readonly List<EventLog> _entries = new List<EventLog>();

        private void Start()
        {
            _sessionStartTime = Time.time;

            string dir = Path.Combine(Application.persistentDataPath, outputFolder);
            Directory.CreateDirectory(dir);

            string timestamp = DateTime.Now.ToString("yyyyMMdd_HHmmss");
            _filePath = Path.Combine(dir, $"log_{participantId}_{timestamp}.csv");

            // CSV header
            File.WriteAllText(_filePath, EventLog.GetCsvHeaders() + "\n");

            Debug.Log($"[ExperimentLogger] Logging to: {_filePath}");

            // Auto-subscribe to gaze events
            if (gazeRaycaster != null)
            {
                gazeRaycaster.onStimulusFocused.AddListener(OnGazeEnter);
                gazeRaycaster.onStimulusUnfocused.AddListener(OnGazeExit);
            }
            else
            {
                Debug.LogError($"[ExperimentLogger] No gaze raycaster found.");
            }
            
            // Auto-subscribe to leap hand tracking events
            if (pinchDetector != null)
            {
                pinchDetector.onPinchDetected.AddListener(OnPinchDetected);
                pinchDetector.onStimulusClicked.AddListener(OnStimulusSelected);
            }
            else
            {
                Debug.LogError($"[ExperimentLogger] No pinch detector found.");
            }
            
            // Stream setup
            var hash = new Hash128();
            hash.Append(StreamName);
            hash.Append(StreamType);
            hash.Append(""+gameObject.GetEntityId());
            StreamInfo streamInfo = new StreamInfo(StreamName, StreamType, 1, LSL.LSL.IRREGULAR_RATE,
                channel_format_t.cf_string, hash.ToString());
            outlet = new StreamOutlet(streamInfo);
        }

        public void OnGazeEnter(Interactable stimulus)
        {
            RecordEvent(new GazeLog("GAZE_ENTER", stimulus));
        }

        public void OnGazeExit(Interactable stimulus)
        {
            RecordEvent(new GazeLog("GAZE_EXIT", stimulus));
        }

        public void OnPinchDetected()
        {
            RecordEvent(new PinchLog());
        }

        public void OnStimulusSelected(Interactable stimulus)
        {
            RecordEvent(new PinchLog(stimulus));
        }

        public void RecordEvent(EventLog log)
        {
            // Gaze data from the raycaster
            Vector3 gazeOrigin = Vector3.zero;
            Vector3 gazeDir = Vector3.forward;

            if (gazeRaycaster != null)
            {
                gazeOrigin = gazeRaycaster.CurrentGazeOrigin;
                gazeDir = gazeRaycaster.CurrentGazeDirection;
            }

            // Head pose (always from the camera transform)
            Transform cam = gazeRaycaster != null ? gazeRaycaster.transform : Camera.main.transform;
            Vector3 headPos = cam.position;
            Vector3 headFwd = cam.forward;

            var defaultData = new EventLog.DefaultLogData
            {
                GazeOrigin = gazeOrigin,
                GazeDirection = gazeDir,
                HeadPosition = headPos,
                HeadForward = headFwd
            };
            log.AddDefaultData(defaultData);
            _entries.Add(log);

            try
            {
                log.LogToConsole();
                File.AppendAllText(_filePath, log.ToCsvEntry() + "\n");
                
                sample[0] = log.ToLslEntry();
                outlet.push_sample(sample);
            }
            catch (Exception e)
            {
                Debug.LogError($"[ExperimentLogger] {e.Message}");
            }
        }

        private void OnDestroy()
        {
            Debug.Log($"[ExperimentLogger] Session ended. {_entries.Count} events recorded to {_filePath}");
        }
    }
}