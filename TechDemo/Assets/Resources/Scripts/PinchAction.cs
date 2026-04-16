using UnityEngine;
using Leap;
using LSL;
using TMPro;
using UnityEngine.UI;
using PinchDetector = Resources.Scripts.PinchDetector;

public class PinchAction : MonoBehaviour
{

    public TMP_Text leapStateText;
    public PinchDetector pinchDetector;
    private LeapServiceProvider leapProvider;

    string StreamName = "LSL4Unity.TechDemo.PinchEvent";
    string StreamType = "Markers";
    private StreamOutlet outlet;
    private string[] sample = { "" };
    private bool isTrackingLH = false;
    private bool isTrackingRH = false;


    // Start is called once before the first execution of Update after the MonoBehaviour is created
    void Start()
    {
        leapProvider = FindAnyObjectByType<LeapServiceProvider>();
        leapProvider.OnUpdateFrame += OnUpdateFrame;
        
        var hash = new Hash128();
        hash.Append(StreamName);
        hash.Append(StreamType);
        hash.Append(""+gameObject.GetEntityId());
        StreamInfo streamInfo = new StreamInfo(StreamName, StreamType, 1, LSL.LSL.IRREGULAR_RATE,
            channel_format_t.cf_string, hash.ToString());
        outlet = new StreamOutlet(streamInfo);
    }

    private void OnUpdateFrame(Frame frame)
    {
        bool newLHState = frame.Hands.Exists(hand => hand.IsLeft);
        bool newRHState = frame.Hands.Exists(hand => !hand.IsLeft);
        
        if (newLHState && !isTrackingLH)   OnHandFound("Left");
        if (!newLHState && isTrackingLH)   OnHandLost("Left");

        if (newRHState && !isTrackingRH) OnHandFound("Right");
        if (!newRHState && isTrackingRH) OnHandLost("Right");

        if (newLHState != isTrackingLH || newRHState != isTrackingRH)
        {
            isTrackingLH = newLHState;
            isTrackingRH = newRHState;
            UpdateText();
        }
    }
    
    private void OnHandFound(string handName)
    {
        Debug.Log($"{handName} hand found");
        sample[0] = $"{handName}HandFound " + gameObject.GetEntityId();
        outlet.push_sample(sample);
    }

    private void OnHandLost(string handName)
    {
        Debug.Log($"{handName} hand lost");
        sample[0] = $"{handName}HandFound " + gameObject.GetEntityId();
        outlet.push_sample(sample);
    }

    private void UpdateText()
    {
        leapStateText.text = $"Left hand: {isTrackingLH} | Right hand: {isTrackingRH}";
    }

    // Update is called once per frame
    void Update()
    {

    }

    private void OnEnable()
    {
        // pinchDetector.OnPinchStart += HandlePinchStart;
    }

    private void OnDisable()
    {
        // pinchDetector.OnPinchStart -= HandlePinchStart;
    }

    private void HandlePinchStart(Hand hand)
    {
        sample[0] = "PinchStart " + gameObject.GetEntityId();
        outlet.push_sample(sample);
    }
}
