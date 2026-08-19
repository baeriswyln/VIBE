using System;
using System.Collections;
using System.Collections.Generic;
using Unity.InferenceEngine.TTS.Inference;
using UnityEngine;
using UnityEngine.Events;
using UnityEngine.InputSystem.Controls;

namespace Experiments
{
    /// <summary>
    /// Base class for all experiments. Inherit from this to create new experiment types.
    /// The ExperimentRunner drives experiments through their lifecycle:
    ///   OnBegin() → Run() coroutine → OnEnd()
    /// </summary>
    public abstract class Experiment : MonoBehaviour
    {
        public ExperimentConfig config;
        public InteractableManager spawner;


        [Header("Events")] public UnityEvent onExperimentStart;
        public UnityEvent onExperimentComplete;
        public bool skipEndOfExperimentPause = false;

        /// <summary>True while this experiment is actively running.</summary>
        public bool IsRunning { get; private set; }

        /// <summary>True when this experiment has finished.</summary>
        public bool IsComplete { get; private set; }

        [HideInInspector] public bool endScheduled = false;

        public virtual void SetConfig(ExperimentConfig config)
        {
            this.config = config;
        }

        public void SetSpawner(InteractableManager spawner)
        {
            this.spawner = spawner;            
        }


        /// <summary>
        /// Called by the runner before Run(). Override for one-time setup.
        /// </summary>
        public virtual void OnBegin()
        {
            IsRunning = true;
            IsComplete = false;
            endScheduled = false;
            onExperimentStart?.Invoke();
        }

        /// <summary>
        /// The main experiment logic. Must be implemented by subclasses.
        /// Runs as a coroutine so you can use yield return to wait for things.
        /// </summary>
        public abstract IEnumerator Run();

        /// <summary>
        /// Called by the runner after Run() completes. Override for cleanup.
        /// </summary>
        public virtual void OnEnd()
        {
            IsRunning = false;
            IsComplete = true;
            onExperimentComplete?.Invoke();
        }

        /// <summary>
        /// Mark this experiment as finished (call from within Run() or externally).
        /// </summary>
        protected void MarkComplete()
        {
            IsComplete = true;
        }
    }

    public enum ExperimentType
    {
        Static,
        Random,
    }

    [System.Serializable]
    public class ExperimentConfig
    {
        public ExperimentType type;
        public int numberOfStimuli;
        public int numberOfTrials = 12;
        public bool hasVirtualHand = true;
        public bool hasVisualFeedback = true;
        public bool hasAuditoryFeedback = true;
        public int numberOfErrors = 0;
        public bool pinchIsHinted = false;
        public int numberOfObserveStimuli = 0;
        public bool isChoiceStimuli = false;

        public ExperimentConfig(ExperimentType type, int numberOfStimuli, int numberOfTrials, 
            bool hasVirtualHand = true, bool hasVisualFeedback = true, bool hasAuditoryFeedback = true, 
            int numberOfErrors = 0, bool pinchIsHinted = false, int numberOfObserveStimuli = 0, 
            bool isChoiceStimuli = false)
        {
            this.type = type;
            this.numberOfStimuli = numberOfStimuli;
            this.numberOfTrials = numberOfTrials;
            this.hasVirtualHand = hasVirtualHand;
            this.hasVisualFeedback = hasVisualFeedback;
            this.hasAuditoryFeedback = hasAuditoryFeedback;
            this.numberOfErrors = numberOfErrors;
            this.pinchIsHinted = pinchIsHinted;
            this.numberOfObserveStimuli = numberOfObserveStimuli;
            this.isChoiceStimuli = isChoiceStimuli;
            
            FixValues();
        }

        public Experiment CreateExperiment(GameObject experimentParent, string name)
        {
            FixValues();
            
            var go = new GameObject(name);
            go.transform.SetParent(experimentParent.transform, worldPositionStays: false);
            go.transform.localPosition = Vector3.zero;
            go.transform.localRotation = Quaternion.identity;

            Experiment comp = null;
            InteractableManager spawner = null;

            switch (type)
            {
                case ExperimentType.Static:
                    comp = go.AddComponent<ExperimentStatic>();
                    spawner = ExperimentManager.Instance.staticSpawner;
                    break;
                case ExperimentType.Random:
                    comp = go.AddComponent<ExperimentRandom>();
                    spawner = ExperimentManager.Instance.randomSpawner;
                    break;
            }

            comp.SetSpawner(spawner);
            comp.SetConfig(this);
            return comp;
        }

        public void CreateExperimentFromSelfAndAddToList(List<Experiment> list, GameObject experimentParent)
        {
            list.Add(CreateExperiment(experimentParent, $"Experiment{list.Count}"));
        }

        private void FixValues()
        {
            numberOfStimuli = Math.Max(numberOfStimuli, 0);
            numberOfErrors = Math.Min(Math.Max(numberOfErrors, 0), numberOfStimuli);
            numberOfObserveStimuli = Math.Min(Math.Max(numberOfObserveStimuli, 0), numberOfStimuli);
        }
    }
}