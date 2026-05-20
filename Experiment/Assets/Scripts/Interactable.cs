using Newtonsoft.Json;
using UnityEngine;
using UnityEngine.Events;

public class Interactable : MonoBehaviour
{
    [Header("Visual Feedback")]
    [Tooltip("Color to apply when the stimulus is highlighted.")]
    public Color highlightColor = new Color(1f, 0.9f, 0.3f, 1f);
        
    [Tooltip("Color to apply when the user gazes at this stimulus.")]
    public Color focusColor = new Color(1f, 0.8f, 0.5f, 1f);
 
    [Tooltip("Color to apply when the stimulus is selected (clicked).")]
    public Color selectedColor = new Color(0.3f, 1f, 0.5f, 1f);
 
    [Tooltip("Scale multiplier when gazed at.")]
    public float highlightScaleMultiplier = 1.15f;
 
    [Tooltip("Speed of the highlight scale animation.")]
    public float scaleAnimationSpeed = 8f;

    [Header("Events")]
    public UnityEvent onGazeEnter;
    public UnityEvent onGazeExit;
    public UnityEvent onSelected;
 
    // State
    private bool _isGazed;
    private bool _isSelected;
    private bool _isHighlighted;
    private Vector3 _originalScale;
    private Color _originalColor;
    private Renderer _renderer;
    private MaterialPropertyBlock _propBlock;
        
    private InteractableManager _interactableManager;
        
    /// <summary>Unique ID assigned at spawn time for logging/experiments.</summary>
    public string StimulusId { get; set; }
 
    public bool IsGazed => _isGazed;
    public bool IsSelected => _isSelected;
    public bool IsHighlighted => _isHighlighted;

    void Awake()
    {
        _originalScale = transform.localScale;
        _renderer = GetComponentInChildren<Renderer>();
        _propBlock = new MaterialPropertyBlock();
 
        if (_renderer != null)
        {
            _renderer.GetPropertyBlock(_propBlock);
            _originalColor = _renderer.sharedMaterial.color;
        }
 
        if (string.IsNullOrEmpty(StimulusId))
            StimulusId = gameObject.name;
    }

    public void Highlight()
    {
        _isHighlighted = true;
        UpdateColor();
    }

    public void SetInteractableManager(InteractableManager interactableManager)
    {
        _interactableManager = interactableManager;
    }

    /// <summary>Called by GazeRaycaster when gaze enters this object.</summary>
    public void OnGazeEnter()
    {
        _isGazed = true;
        UpdateColor();
        onGazeEnter?.Invoke();
    }
 
    /// <summary>Called by GazeRaycaster when gaze exits this object.</summary>
    public void OnGazeExit()
    {
        _isGazed = false;
 
        // Revert to selected color or original
        UpdateColor();
        onGazeExit?.Invoke();
    }
 
    /// <summary>Called by PinchDetector when the user performs a pinch while gazing.</summary>
    public void OnSelect()
    {
        _isSelected = true;
        UpdateColor();
        onSelected?.Invoke();
    }
 
    /// <summary>Resets this stimulus to its default visual state.</summary>
    public void ResetState()
    {
        _isSelected = false;
        _isGazed = false;
        _isHighlighted = false;
        UpdateColor();
        transform.localScale = _originalScale;
    }
 
    private void UpdateColor()
    {
        var color = _originalColor;
        color = _isHighlighted ? highlightColor : color;
        color = _isGazed ? focusColor : color;
        color = _isSelected ? selectedColor : color;
            
            
        if (_renderer == null) return;
 
        _renderer.GetPropertyBlock(_propBlock);
        _propBlock.SetColor("_Color", color);
        _propBlock.SetColor("_BaseColor", color); // URP/HDRP
        _renderer.SetPropertyBlock(_propBlock);
    }

    public string ToJSON()
    {
        var pos = transform.position;
        return JsonConvert.SerializeObject(new
        {
            StimulusId, IsHighlighted, IsSelected, IsGazed,
            Position = new { x = pos.x, y = pos.y, z = pos.z },
        });
    }
}