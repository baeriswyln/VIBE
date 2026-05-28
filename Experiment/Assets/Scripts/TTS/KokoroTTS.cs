using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using Newtonsoft.Json.Linq;
using Unity.InferenceEngine.TTS.Assets;
using UnityEngine;

namespace Unity.InferenceEngine.TTS.Inference
{
    [RequireComponent(typeof(AudioSource))]
    public class KokoroTTS : MonoBehaviour
    {
        [Header("Asset paths (relative to StreamingAssets)")]
        [SerializeField] string modelPath = "Kokoro/model.onnx";
        [SerializeField] string voicePath = "Kokoro/voices/af_heart.bin";
        [SerializeField] string vocabPath = "Kokoro/vocab.json";

        [Header("Inference")]
        [SerializeField] BackendType backend = BackendType.GPUCompute;
        [SerializeField, Range(0.5f, 2f)] float speed = 1f;

        const int SAMPLE_RATE = 24_000;
        const int STYLE_DIM   = 256;
        const int MAX_TOKENS  = 510; // leaves room for pad 0s on both sides

        Model      _model;
        Worker     _worker;
        float[]    _voiceData;      // flat, length = N * STYLE_DIM
        int        _voiceMaxTokens; // N
        Dictionary<char,int> _vocab;
        AudioSource _audio;
        
        // Signleton stuff
        public static KokoroTTS Instance { get;  private set; } 

        void Awake()
        {
            if (Instance != null && Instance != this)
            {
                Destroy(gameObject);
                return;
            }
            Instance = this;
            DontDestroyOnLoad(gameObject);
            
            _audio = GetComponent<AudioSource>();

            // --- Load the ONNX model ---
            var modelAsset = UnityEngine.Resources.Load<ModelAsset>(modelPath);
            _model = ModelLoader.Load(modelAsset); 
            _worker = new Worker(_model, backend);

            // --- Load voice .bin (flat float32 array) ---
            var voiceAsset = UnityEngine.Resources.Load<RawBytesAsset>(voicePath);
            var voiceBytes = voiceAsset.bytes;
            _voiceData = new float[voiceBytes.Length / 4];
            Buffer.BlockCopy(voiceBytes, 0, _voiceData, 0, voiceBytes.Length);
            _voiceMaxTokens = _voiceData.Length / STYLE_DIM;

            // --- Load phoneme→id vocab ---
            var vocabJson = UnityEngine.Resources.Load<TextAsset>(vocabPath);
            _vocab = ParseVocab(vocabJson.text);
        }

        void OnDestroy()
        {
            _worker?.Dispose();
        }

        /// <summary>
        /// Synthesize and play. `phonemes` is an already-phonemized string,
        /// e.g. "həlˈoʊ wˈɜːld" — characters in the Kokoro phoneme vocabulary.
        /// </summary>
        public async Task SpeakAsync(string text)
        {
            // 0) text -> phoneme chars
            var phonemes = MisakiSharp.TextToPhonemes(text);
            
            // 1) phoneme chars → token ids
            var ids = new List<int>(phonemes.Length);
            foreach (var c in phonemes)
                if (_vocab.TryGetValue(c, out var id)) ids.Add(id);

            if (ids.Count == 0) { Debug.LogWarning("No recognized phonemes."); return; }
            if (ids.Count > MAX_TOKENS) ids = ids.GetRange(0, MAX_TOKENS);

            // 2) pick the style row keyed by the *unpadded* token count
            int row = Mathf.Clamp(ids.Count, 0, _voiceMaxTokens - 1);
            var style = new float[STYLE_DIM];
            Array.Copy(_voiceData, row * STYLE_DIM, style, 0, STYLE_DIM);

            // 3) prepend/append pad token 0
            var padded = new long[ids.Count + 2];
            padded[0] = 0;
            for (int i = 0; i < ids.Count; i++) padded[i + 1] = ids[i];
            padded[^1] = 0;

            // 4) build the three input tensors
            // input_ids: int64, shape [1, T]
            using var inputIds = new Tensor<int>(new TensorShape(1, padded.Length),
                padded.Select(x => (int)x).ToArray());
            // (Kokoro's ONNX declares int64; Inference Engine's int-typed tensor is fine
            //  if your imported model accepts Int32. If you hit a type-mismatch on import,
            //  the sample shows how to add a cast layer with the Functional API. See the
            //  "If the model expects int64" note below.)

            using var styleTensor = new Tensor<float>(new TensorShape(1, STYLE_DIM), style);
            using var speedTensor = new Tensor<float>(new TensorShape(1), new[] { speed });

            // 5) feed inputs by name and schedule
            _worker.SetInput("input_ids", inputIds);
            _worker.SetInput("style",     styleTensor);
            _worker.SetInput("speed",     speedTensor);
            // _worker.Schedule();
            var schedule = _worker.ScheduleIterable();
            int step = 0;
            while (schedule.MoveNext())
            {
                // dispatch a few layers per frame, then let the engine breathe
                if (++step % 40 == 0)
                    await Task.Yield();   // returns control to main loop, resumes next frame
            }
            
            // 6) read back the waveform asynchronously
            var output = _worker.PeekOutput() as Tensor<float>;
            using var cpu = await output.ReadbackAndCloneAsync();
            var pcm = cpu.DownloadToArray(); // float[] length = number of samples

            // 7) hand to AudioSource
            var clip = AudioClip.Create("KokoroOutput", pcm.Length, 1, SAMPLE_RATE, false);
            clip.SetData(pcm, 0);
            _audio.clip = clip;
            _audio.Play();
        }

        public IEnumerator WaitForEndOfPlayback()
        {
            yield return new WaitForSeconds(0.3f);
            while (_audio.isPlaying)
            {
                yield return new WaitForSeconds(0.1f);
            }
            yield return new WaitForSeconds(0.3f);
        }

        public IEnumerator SpeakAndWait(string text)
        {
            var task = SpeakAsync(text);
            yield return new WaitUntil(() => task.IsCompleted);
            yield return WaitForEndOfPlayback();
        }

        static Dictionary<char,int> ParseVocab(string json)
        {
            // The vocab in Kokoro's config.json is a flat object like
            // { "$": 0, ";": 1, ":": 2, ... }. 

            var root = JObject.Parse(json);
            var vocabObj = (JObject)root["model"]["vocab"];
            
            // Deserialize as string keys first, then convert to char keys
            var stringVocab = vocabObj.ToObject<Dictionary<string, int>>();

            var charVocab = new Dictionary<char, int>(stringVocab.Count);
            foreach (var kvp in stringVocab)
            {
                if (kvp.Key.Length != 1)
                {
                    Debug.LogWarning($"Skipping non-single-character token: \"{kvp.Key}\" (length {kvp.Key.Length})");
                    continue;
                }
                charVocab[kvp.Key[0]] = kvp.Value;
            }

            return charVocab;
        }
    }
}