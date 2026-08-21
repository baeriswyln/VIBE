using System;
using Experiments;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;
using UnityEngine;

namespace Logging
{
    public abstract class EventLog
    {
        public enum LogSeverity
        {
            MESSAGE,
            WARNING,
            ERROR
        }

        public LogSeverity Severity = LogSeverity.MESSAGE;
        private HeadTrackingData headTrackingData;
        private ExperimentConfig _experimentConfig = ExperimentManager.Instance.CurrentExperiment?.config;
        private int _experimentIndex = ExperimentManager.Instance.CurrentIndex;
        private DateTime timestamp { get; set; } = DateTime.Now;
        private string eventType => GetType().Name;
        protected string eventName;

        public void AddHeadTrackingData(HeadTrackingData data)
        {
            headTrackingData = data;
        }
        

        public string AsJson()
        {
            return JsonConvert.SerializeObject(new
            {
                UnityTimestamp = timestamp.ToString("O"),
                EventType = eventType,
                EventName = eventName,
                Severity = Severity.ToString(),
                ExperimentName = ExperimentManager.Instance.CurrentExperiment?.ToString() ?? String.Empty,
                ExperimentConfig = _experimentConfig,
                ExperimentIndex = _experimentIndex,
                Payload = new JRaw(GetPayload()),
                HeadTracking = headTrackingData
            });
        }

        public void LogToConsole()
        {
            var msg = $"[{eventType}] {eventName}: {GetPayload()}";
            switch (Severity)
            {
                case LogSeverity.MESSAGE:
                    Debug.Log(msg);
                    break;
                case LogSeverity.WARNING:
                    Debug.LogWarning(msg);
                    break;
                case LogSeverity.ERROR:
                    Debug.LogError(msg);
                    break;
            }
        }

        protected abstract string GetPayload();

        public struct HeadTrackingData
        {
            public Vector3 GazeOrigin;
            public Vector3 GazeDirection;
            public Vector3 HeadPosition;
            public Vector3 HeadForward;
        }
    }
}