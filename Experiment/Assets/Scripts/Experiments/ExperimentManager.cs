using System;
using System.Collections;
using System.Collections.Generic;
using System.Diagnostics;
using Logging;
using Newtonsoft.Json;
using TMPro;
using Unity.InferenceEngine.TTS.Inference;
using UnityEngine;
using UnityEngine.Events;
using UnityEngine.InputSystem;
using UnityEngine.Serialization;

namespace Experiments
{
    public class ExperimentManager : MonoBehaviour
    {
        public static ExperimentManager Instance { get; private set; }

        [Header("Experiments Session 1")] 
        public Experiment tutorialSession1;
        public ExperimentConfig controlExperimentSession1;
        public List<ExperimentConfig> experimentConfigsSession1;
        
        [Header("Experiments Session 2")]
        public Experiment tutorialSession2;
        public ExperimentConfig controlExperimentSession2;
        public List<ExperimentConfig> experimentConfigsSession2;
        
        private List<Experiment> _experimentPlaylist;
        
        [Header("General Experiment Configuration")]
        [Tooltip("Seconds to wait between trials.")]
        public float interTrialIntervalMin = .5f;
        public float interTrialIntervalMax = 2.5f;

        [Tooltip("Seconds to show feedback (correct/incorrect) before moving on.")]
        public float feedbackDuration = 0.5f;
        
        [Tooltip("Time required gazing at an interactable to activate the negative highlight condition.")]
        public float gazeMinTime = 0.2f;
        
        [Header("References")] [Tooltip("StimulusSpawner that handles placing the stimuli.")]
        public InteractableManager spawner;

        [Tooltip("Seconds to wait between experiments.")]
        public float gapBetweenExperiments = 1f;
        
        public InputActionAsset inputActionAsset;
        
        [Header("UI Elements")]
        [SerializeField] private GazeRaycaster gazeRaycaster;
        [SerializeField] private GameObject pauseUi;
        [SerializeField] private TMP_Text instructionTextUi;
        [SerializeField] private TMP_Text progress;

        [Header("Events")] public UnityEvent onSequenceStart;
        public UnityEvent onSequenceComplete;
        public UnityEvent<Experiment> onExperimentStarted;
        public UnityEvent<Experiment> onExperimentFinished;

        // State
        private int _currentIndex = -1;
        private bool _isRunning;
        private bool _isPaused = false;

        public bool IsRunning => _isRunning;
        public int CurrentIndex => _currentIndex;

        public Experiment CurrentExperiment =>
            (_currentIndex >= 0 && _currentIndex < _experimentPlaylist.Count)
                ? _experimentPlaylist[_currentIndex]
                : null;

        // Singleton initialization
        private void Awake()
        {
            if (Instance != null && Instance != this)
            {
                Destroy(gameObject);
                return;
            }
            Instance = this;
            DontDestroyOnLoad(gameObject);
        }
        
        private void Start()
        {
        }

        private void OnEnable()
        {
            inputActionAsset.FindActionMap("Experiment").Enable();
            InputSystem.actions.FindAction("ExpNext").performed += SkipCurrent;
            InputSystem.actions.FindAction("ExpPrevious").performed += GoToPrevious;
            InputSystem.actions.FindAction("ExpRestart").performed += RestartCurrent;
            InputSystem.actions.FindAction("ExpPause").performed += TogglePause;
            InputSystem.actions.FindAction("ExpStartSession1").performed += StartSequence1;
            InputSystem.actions.FindAction("ExpStartSession2").performed += StartSequence2;
        }

        private void OnDisable()
        {
            InputSystem.actions.FindAction("ExpStartSession2").performed += StartSequence2;
            InputSystem.actions.FindAction("ExpStartSession1").performed += StartSequence1;
            InputSystem.actions.FindAction("ExpPause").performed -= TogglePause;
            InputSystem.actions.FindAction("ExpRestart").performed -= RestartCurrent;
            InputSystem.actions.FindAction("ExpPrevious").performed -= GoToPrevious;
            InputSystem.actions.FindAction("ExpNext").performed -= SkipCurrent;
            inputActionAsset.FindActionMap("Experiment").Disable();
        }

        public void StartSequence1(InputAction.CallbackContext context)
        {
            StartCoroutine(StartSequence(ExperimentSession.Session1));
        }

        public void StartSequence2(InputAction.CallbackContext context)
        {
            StartCoroutine(StartSequence(ExperimentSession.Session2));
        }

        /// <summary>
        /// Begin running the experiment sequence from the first experiment.
        /// </summary>
        public IEnumerator StartSequence(ExperimentSession session)
        {
            if (_isRunning)
            {
                ExperimentLogger.Instance.Warn(new ExperimentManagerLog("SEQUENCE_ALREADY_RUNNING", ""));
                yield break;
            }

            _experimentPlaylist = new List<Experiment>();

            switch (session)
            {
                case ExperimentSession.Session1:
                    yield return PlayTextPages(new []{"Starting session 1"});
                    _experimentPlaylist.Add(tutorialSession1);
                    controlExperimentSession1.CreateExperimentFromSelfAndAddToList(_experimentPlaylist);
                    foreach (var cfg in experimentConfigsSession1) cfg.CreateExperimentFromSelfAndAddToList(_experimentPlaylist);
                    controlExperimentSession1.CreateExperimentFromSelfAndAddToList(_experimentPlaylist);
                    break;
                case ExperimentSession.Session2:
                    yield return PlayTextPages(new []{"Starting session 1"});
                    _experimentPlaylist.Add(tutorialSession2);
                    controlExperimentSession2.CreateExperimentFromSelfAndAddToList(_experimentPlaylist);
                    foreach (var cfg in experimentConfigsSession2) cfg.CreateExperimentFromSelfAndAddToList(_experimentPlaylist);
                    controlExperimentSession2.CreateExperimentFromSelfAndAddToList(_experimentPlaylist);
                    break;
            }

            yield return RunAllExperiments();
        }

        /// <summary>
        /// Skip the currently running experiment and move to the next one.
        /// Useful for debugging or a skip button.
        /// </summary>
        public void SkipCurrent(InputAction.CallbackContext context)
        {
            if (CurrentExperiment != null)
            {
                ExperimentLogger.Instance.Log(new ExperimentManagerLog("SEQUENCE_SKIP", CurrentExperiment.ToString()));
                CurrentExperiment.endScheduled = true;
            }
        }

        public void RestartCurrent(InputAction.CallbackContext context)
        {
            if (CurrentExperiment != null)
            {
                ExperimentLogger.Instance.Log(new ExperimentManagerLog("SEQUENCE_RESTART", CurrentExperiment.ToString()));
                CurrentExperiment.endScheduled = true;
                _currentIndex -= 1;
            }
        }
        
        public void GoToPrevious(InputAction.CallbackContext context)
        {
            if (CurrentExperiment != null)
            {
                ExperimentLogger.Instance.Log(new ExperimentManagerLog("SEQUENCE_PREVIOUS", CurrentExperiment.ToString()));
                CurrentExperiment.endScheduled = true;
                _currentIndex = Math.Max(-1, _currentIndex-2);
            }
        }

        public void TogglePause(InputAction.CallbackContext context)
        {
            if (CurrentExperiment != null)
            {
                ExperimentLogger.Instance.Log(new ExperimentManagerLog("TOGGLE_PAUSE", CurrentExperiment.ToString()));

                if (_isPaused)
                {
                    Time.timeScale = 1;
                    AudioListener.pause = false;
                    gazeRaycaster.Activate();
                    pauseUi.SetActive(false);
                }
                else
                {
                    Time.timeScale = 0;
                    AudioListener.pause = true;
                    gazeRaycaster.Deactivate();
                    pauseUi.SetActive(true);
                }
                _isPaused = !_isPaused;
            }
        }

        private IEnumerator RunAllExperiments()
        {
            _isRunning = true;
            PinchHandler.Instance.Active = false;
            onSequenceStart?.Invoke();
            ExperimentLogger.Instance.Log(new ExperimentManagerLog("SEQUENCE_START", ""));

            for (_currentIndex = 0; _currentIndex < _experimentPlaylist.Count; _currentIndex++)
            {
                Experiment exp = _experimentPlaylist[_currentIndex];
                if (!exp)
                {
                    ExperimentLogger.Instance.Warn(new ExperimentManagerLog("MISSING_EXPERIMENT", $"Experiment at index {_currentIndex} is null, skipping."));
                    continue;
                }

                // Disable all other experiments' GameObjects so they don't interfere
                SetOnlyActive(exp);

                ExperimentLogger.Instance.Log(new ExperimentManagerLog("EXPERIMENT_START", $"{_currentIndex}:{exp.ToString()}"));
                exp.OnBegin();
                onExperimentStarted?.Invoke(exp);

                yield return StartCoroutine(exp.Run());

                exp.OnEnd();
                ExperimentLogger.Instance.Log(new ExperimentManagerLog("EXPERIMENT_END", $"{_currentIndex}:{exp.ToString()}"));
                onExperimentFinished?.Invoke(exp);

                if (gapBetweenExperiments > 0f && !exp.endScheduled && !exp.skipEndOfExperimentPause)
                    yield return new WaitForSeconds(gapBetweenExperiments);
            }

            _currentIndex = -1;
            _isRunning = false;
            onSequenceComplete?.Invoke();
            ExperimentLogger.Instance.Log(new ExperimentManagerLog("SEQUENCE_COMPLETE", ""));
        }

        public IEnumerator PlayExperimentInstructions(ExperimentConfig config)
        {
            var pages = new List<string>();
            
            if (config.isChoiceStimuli)
                pages.Add("Select the stimulus with the color you like most amongst the colors shown.");
            else
            {
                var text = "Select the highlighted stimulus.";
                if (config.numberOfObserveStimuli > 0) text += " A stimulus highlighted red should only be observed. Look at such a stimulus for approximately half a second without pinching.";
                pages.Add(text);
            }
            
            if (config.pinchIsHinted)
                pages.Add("Pinches should only be hinted in this round. Only think about pinching and execute a small hinted movement without actually pinching.");
            
            yield return PlayTextPages(pages.ToArray());
        }

        public IEnumerator PlayTextPages(string[] pages, bool hideAfterPlay = true)
        {
            if (instructionTextUi) instructionTextUi.gameObject.SetActive(true);
            for (var i = 0; i < pages.Length && (!CurrentExperiment || !CurrentExperiment.endScheduled); i++)
            {
                if(instructionTextUi) instructionTextUi.text = pages[i];
                yield return KokoroTTS.Instance.SpeakAndWait(pages[i]);
            }
            if (instructionTextUi && hideAfterPlay) instructionTextUi.gameObject.SetActive(false);
        }

        /// <summary>
        /// Enables the given experiment's GameObject and disables all others.
        /// </summary>
        private void SetOnlyActive(Experiment active)
        {
            foreach (var exp in _experimentPlaylist)
            {
                if (!exp) continue;
                exp.gameObject.SetActive(exp == active);
            }
        }

        protected class ExperimentManagerLog : EventLog
        {
            private string _message;
            
            public ExperimentManagerLog(string name, string message)
            {
                eventName = name;
                _message = message;
            }

            protected override string GetPayload()
            {
                return JsonConvert.SerializeObject(new
                {
                    Message = _message,
                });
            }
        }
        
        public enum ExperimentSession
        {
            Session1,
            Session2
        }
    }
}