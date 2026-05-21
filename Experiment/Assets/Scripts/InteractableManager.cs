using System.Collections.Generic;
using UnityEngine;

public class InteractableManager : MonoBehaviour
{
    [Header("Spawn properties")]
    [SerializeField] private Interactable stimuliPrefab;
    [SerializeField] private BoxCollider spawnBox;
    [SerializeField] private int maxSpawnAttempts = 5;
    [SerializeField] private float minStimuliGap = 0.5f;
        
    [Header("Curved Spawn Area")]
    public bool useCurvedBounds = true;
    public float innerRadius = 1.5f;       // distance from player
    public float outerRadius = 2.5f;       // optional depth
    public float arcAngleDegrees = 90f;    // total horizontal arc width
    public float minHeight = 0f;       // vertical span
    public float maxHeight = 1.5f;       // vertical span

    private List<Vector3> _spawnedStimuliPositions;
    private List<Interactable> _spawnedStimuli;
    private float _prefabRadius;

    public List<Interactable> spawnedStimuli => _spawnedStimuli;

    // Start is called once before the first execution of Update after the MonoBehaviour is created
    void Start()
    {
        var sphere = stimuliPrefab.GetComponentInChildren<SphereCollider>();
        var scale = sphere.transform.lossyScale;
        var maxScale = Mathf.Max(scale.x, scale.y, scale.z);
        _prefabRadius = sphere.radius * maxScale;
    }

    // Update is called once per frame
    void Update()
    {
    }

    public void SpawnStimuli(int nbStimuli)
    {
        _spawnedStimuli = new List<Interactable>();
        _spawnedStimuliPositions = new List<Vector3>();

        for (int i = 0; i < nbStimuli; i++)
        {
            Vector3? position = TryGetValidPosition(_spawnedStimuliPositions);

            if (position.HasValue)
            {
                var stimuli = Instantiate(stimuliPrefab, position.Value, Quaternion.identity);
                stimuli.SetInteractableManager(this);
                _spawnedStimuli.Add(stimuli);
                _spawnedStimuliPositions.Add(position.Value);
            }
            else
            {
                Debug.LogWarning($"Could not find valid position for prefab {i}");
            }
        }
    }
    
    public void SpawnStimulusInCircle(int count, Vector3 center, float radius)
    {
        _spawnedStimuli = new List<Interactable>();
        _spawnedStimuliPositions = new List<Vector3>();

        if (count <= 0)
            return;

        // Special case: a single object goes at the center.
        if (count == 1)
        {
            var stimulus = Instantiate(stimuliPrefab, center, Quaternion.identity);
            _spawnedStimuli.Add(stimulus);
            _spawnedStimuliPositions.Add(center);
            return;
        }

        // Distribute evenly around the circle.
        // Start angle of 90° puts the first object at the top, and going
        // clockwise gives the intuitive left/right and top/bottom layouts.
        float angleStep = 360f / count;

        for (int i = 0; i < count; i++)
        {
            float angleDeg = i * angleStep;
            float angleRad = angleDeg * Mathf.Deg2Rad;

            Vector3 offset = new Vector3(
                Mathf.Cos(angleRad) * radius,
                Mathf.Sin(angleRad) * radius,
                0f
            );

            var stimulus = Instantiate(stimuliPrefab, center+offset, Quaternion.identity);
            _spawnedStimuli.Add(stimulus);
            _spawnedStimuliPositions.Add(center+offset);
        }
    }

    public void DestroyStimuli()
    {
        foreach (var stimuli in _spawnedStimuli)
        {
            Destroy(stimuli.gameObject);
        }
        _spawnedStimuli = new List<Interactable>();
        _spawnedStimuliPositions = new List<Vector3>();
    }

    public void ResetAll()
    {
        foreach (var stimuli in _spawnedStimuli)
        {
            stimuli.ResetState();
        }
    }

    Vector3? TryGetValidPosition(List<Vector3> existing)
    {
        for (int attempt = 0; attempt < maxSpawnAttempts; attempt++)
        {
            Vector3 candidate = useCurvedBounds ? GetRandomPosition() : GetRandomPositionInBox();

            if (!OverlapsAny(candidate, existing))
                return candidate;
        }

        return null;
    }

    Vector3 GetRandomPositionInBox()
    {
        Bounds bounds = spawnBox.bounds;

        return new Vector3(
            Random.Range(bounds.min.x, bounds.max.x),
            Random.Range(bounds.min.y, bounds.max.y),
            Random.Range(bounds.min.z, bounds.max.z)
        );
    }

    Vector3 GetRandomPosition()
    {
        float angle = Random.Range(-arcAngleDegrees / 2f, arcAngleDegrees / 2f) * Mathf.Deg2Rad;
        float radius = Random.Range(innerRadius, outerRadius);
        float height = Random.Range(minHeight, maxHeight);
            
        return new Vector3(
            Mathf.Sin(angle) * radius,
            height,
            Mathf.Cos(angle) * radius
        );
    }

    bool OverlapsAny(Vector3 candidate, List<Vector3> existing)
    {
        foreach (var pos in existing)
        {
            if (Vector3.Distance(candidate, pos) < _prefabRadius * 2f + minStimuliGap)
                return true;
        }

        return false;
    }

    float GetPrefabRadius()
    {
        Collider col = stimuliPrefab.GetComponent<Collider>();
        if (col != null) return col.bounds.extents.magnitude;

        Renderer rend = stimuliPrefab.GetComponent<Renderer>();
        if (rend != null) return rend.bounds.extents.magnitude;

        return 0.5f;
    }
}