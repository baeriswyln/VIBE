namespace Logging
{
    public class GazeLog : EventLog
    {
        private Interactable _interactable;
        
        public GazeLog(string name, Interactable stimuli)
        {
            eventName = name;
            _interactable = stimuli;
        }

        protected override object GetPayload()
        {
            return _interactable.ToLoggableObject();
        }
    }
}