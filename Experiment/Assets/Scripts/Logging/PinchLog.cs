namespace Logging
{
    public class PinchLog : EventLog
    {
        private Interactable _interactable;
        
        public PinchLog()
        {
            eventName = "PINCH_NONE";
        }

        public PinchLog(Interactable stimulus)
        {
            eventName = "PINCH_STIMULUS";
            _interactable = stimulus;
        }

        protected override string GetPayload()
        {
            if (_interactable)
            {
                return _interactable.ToJSON();
            }
            else
            {
                return "{}";
            }
        }
    }
}