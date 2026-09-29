namespace Logging
{
    public class PinchLog : EventLog
    {
        public enum PinchType
        {
            PINCH,
            PINCH_IGNORED,
            PINCH_WIZARD_OF_OZ,
            GAZE,
            IGNORED_GROUND_TRUTH,
            IGNORED_EEG_PINCH,
        }
        
        private Interactable _interactable;
        private PinchType _pinchType;
        
        public PinchLog(PinchType pinchType = PinchType.PINCH)
        {
            eventName = "PINCH_NONE";
            _pinchType = pinchType;
        }

        public PinchLog(Interactable stimulus, PinchType pinchType = PinchType.PINCH)
        {
            eventName = "PINCH_STIMULUS";
            _interactable = stimulus;
            _pinchType = pinchType;
        }

        protected override object GetPayload()
        {
            return new
            {
                Interactable = _interactable?.ToLoggableObject() ?? new {},
                PinchType = _pinchType.ToString(),
            };
        }
    }
}