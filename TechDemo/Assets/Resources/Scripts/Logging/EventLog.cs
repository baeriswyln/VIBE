using System;
using UnityEngine;

namespace Resources.Scripts.Logging
{
    public abstract class EventLog
    {
        public enum LogSeverity
        {
            MESSAGE, WARNING, ERROR
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
            => new[] { "Timestamp", "Severity", "EventType", "EventName", "Payload"};

        public string[] ToCsvEntry()
            => new[] { timestamp.ToString("O"), Severity.ToString(), eventType, GetPayload()};

        public string ToLslEntry()
        {
            return "";
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