using UnityEngine;
using UnityEngine.InputSystem;

/// <summary>
/// First-person mouse look for testing the experiment without a VR headset.
/// Added automatically (and only) when no XR device is active — see NonVRTestModeBootstrap.
/// Rotates its own transform, so it can sit directly on the same GameObject
/// that GazeRaycaster casts its gaze ray from.
/// </summary>
public class MouseLookController : MonoBehaviour
{
    [Tooltip("Mouse look sensitivity (degrees per pixel of mouse delta).")]
    public float sensitivity = 0.1f;

    [Tooltip("Minimum/maximum vertical look angle.")]
    public float minPitch = -80f;
    public float maxPitch = 80f;

    [Tooltip("Lock and hide the cursor while in mouse-look mode. Press Escape to release it.")]
    public bool lockCursor = true;

    private float _yaw;
    private float _pitch;

    private void Start()
    {
        Vector3 e = transform.localEulerAngles;
        _yaw = e.y;
        _pitch = e.x > 180f ? e.x - 360f : e.x;

        SetCursorLocked(lockCursor);
    }

    private void Update()
    {
        if (Keyboard.current != null && Keyboard.current.escapeKey.wasPressedThisFrame)
        {
            SetCursorLocked(Cursor.lockState != CursorLockMode.Locked);
        }

        if (Cursor.lockState != CursorLockMode.Locked || Mouse.current == null)
            return;

        Vector2 delta = Mouse.current.delta.ReadValue();
        _yaw += delta.x * sensitivity;
        _pitch = Mathf.Clamp(_pitch - delta.y * sensitivity, minPitch, maxPitch);

        transform.localRotation = Quaternion.Euler(_pitch, _yaw, 0f);
    }

    private void SetCursorLocked(bool locked)
    {
        Cursor.lockState = locked ? CursorLockMode.Locked : CursorLockMode.None;
        Cursor.visible = !locked;
    }
}