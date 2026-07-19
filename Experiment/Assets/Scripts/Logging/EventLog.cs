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
        private DefaultLogData logData;
        private DateTime timestamp { get; set; }
        private string eventType => GetType().Name;
        protected string eventName;

        protected EventLog()
        {
            timestamp = DateTime.Now;
        }

        public void AddDefaultData(DefaultLogData data)
        {
            logData = data;
        }

        public static string[] GetCsvHeaders()
            => new[] { "Timestamp", "Severity", "EventType", "EventName", "ExperimentName", "Payload" };

        public string[] ToCsvEntry()
            => new[] { timestamp.ToString("O"), Severity.ToString(), eventType,
                ExperimentManager.Instance.CurrentExperiment?.ToString() ?? String.Empty,
                GetPayload() };

        public string ToLslEntry()
        {
            return JsonConvert.SerializeObject(new
            {
                UnityTimestamp = timestamp.ToString("O"),
                EventType = eventType,
                EventName = eventName,
                Severity = Severity.ToString(),
                ExperimentName = ExperimentManager.Instance.CurrentExperiment?.ToString() ?? String.Empty,
                Payload = new JRaw(GetPayload()),
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

        public struct DefaultLogData
        {
            public Vector3 GazeOrigin;
            public Vector3 GazeDirection;
            public Vector3 HeadPosition;
            public Vector3 HeadForward;
        }
    }
}