using Newtonsoft.Json;

namespace Logging
{
    public class PinchLog : EventLog
    {
        public enum PinchType
        {
            PINCH,
            PINCH_IGNORED,
            PINCH_WIZARD_OF_OZ,
            GAZE
        }
        
        private Interactable _interactable;
        private PinchType _pinchType;
        
        public PinchLog()
        {
            eventName = "PINCH_NONE";
            _pinchType = PinchType.PINCH;
        }

        public PinchLog(Interactable stimulus, PinchType pinchType = PinchType.PINCH)
        {
            eventName = "PINCH_STIMULUS";
            _interactable = stimulus;
            _pinchType = pinchType;
        }

        protected override string GetPayload()
        {
            return JsonConvert.SerializeObject(new
            {
                Interactable = _interactable,
                PinchType = _pinchType,
            });
        }
    }
}