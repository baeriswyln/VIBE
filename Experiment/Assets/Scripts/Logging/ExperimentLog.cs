using Newtonsoft.Json;

namespace Logging
{
    public class ExperimentLog : EventLog
    {
        private string _message; 

        public ExperimentLog(string experimentName, string message)
        {
            eventName = experimentName;
            _message = message;
        }
        
        public ExperimentLog(string experimentName, string message, LogSeverity severity)
        {
            eventName = experimentName;
            _message = message;
            base.Severity = severity;
        }

        protected override object GetPayload()
        {
            return new
            {
                Message = _message
            };
        }
    }
}