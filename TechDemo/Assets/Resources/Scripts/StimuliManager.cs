using UnityEngine;
using UnityEngine.UI;
using LSL;

public class StimuliManager : MonoBehaviour
{
    // Start is called once before the first execution of Update after the MonoBehaviour is created

    public Image colorOutput;

    private float timePassed = 0f;

    string StreamName = "LSL4Unity.TechDemo.StimuliEvent";
    string StreamType = "Markers";
    private StreamOutlet outlet;
    private string[] sample = { "" };

    void Start()
    {
        var hash = new Hash128();
        hash.Append(StreamName);
        hash.Append(StreamType);
        hash.Append(""+gameObject.GetEntityId());
        StreamInfo streamInfo = new StreamInfo(StreamName, StreamType, 1, LSL.LSL.IRREGULAR_RATE,
            channel_format_t.cf_string, hash.ToString());
        outlet = new StreamOutlet(streamInfo);
    }

    // Update is called once per frame
    void Update()
    {
        timePassed += Time.deltaTime;

        if (timePassed >= 1f)
        {
            colorOutput.color = new Color(Random.value, Random.value, Random.value);
            sample[0] = "ColorChange " + gameObject.GetEntityId();
            outlet.push_sample(sample);
            timePassed = 0f;
        }
    }
}
