using System.Collections;
using System.Collections.Generic;
using Logging;
using UnityEngine;

namespace Experiments
{
    public class ExperimentStatic : StimuliExperiment
    {
        [Header("Stimulus Spawn Configuration")]
        public Vector3 spawnCenter = new(0, 1, 5);
        public float spawnRadius;
        public bool respawnBetweenRounds = true;
        
        public override IEnumerator Run()
        {
            if (!spawner)
            {
                ExperimentLogger.Instance.Error(new ExperimentLog(GetType().Name, "No spawner assigned."));
                yield break;
            }

            yield return PlayTextPages(textStart);
            if(trialInfoText) trialInfoText.gameObject.SetActive(false);
            
            spawner.SpawnStimulusInCircle(stimuliPerTrial, spawnCenter, spawnRadius);

            for (var i = 0; i < numberOfTrials; i++)
            {
                yield return RunSingleTrial(i);
                if(endScheduled)
                {
                    spawner.DestroyStimuli();
                    yield break;
                }
                if(respawnBetweenRounds) spawner.DestroyStimuli();
                yield return WaitForNextStimulus();
                if(respawnBetweenRounds) spawner.SpawnStimulusInCircle(stimuliPerTrial, spawnCenter, spawnRadius);
            }
            
            spawner.DestroyStimuli();
            yield return PlayTextPages(textEnd);
            ExperimentLogger.Instance.Log(new ExperimentLog(GetType().Name, $"Finished: {_results.Count} events."));
            MarkComplete();
        }
    }
}