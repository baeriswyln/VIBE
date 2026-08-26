using System.Collections;
using Logging;
using UnityEngine;

namespace Experiments
{
    public class ExperimentEmpty : Experiment
    
    
    {
        public override IEnumerator Run(bool skipEndPause = false)
        {
            // yield return PlayTextPages(textStart);
            // if(trialInfoText) trialInfoText.gameObject.SetActive(false);
            if(endScheduled) yield break;

            // yield return PlayTextPages(textEnd);
            ExperimentLogger.Instance.Log(new ExperimentLog(ToString(), $"Finished."));
            MarkComplete();
        }
    }
}