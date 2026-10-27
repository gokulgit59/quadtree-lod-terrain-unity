using UnityEngine;

/// <summary>
/// First-person walker for the terrain.
///   WASD move, Shift sprint, Space jump, mouse look.
///   Click the Game view to capture the mouse, Esc to release it.
///
/// Put it on a GameObject with a CharacterController and make the camera a child.
/// </summary>
[RequireComponent(typeof(CharacterController))]
public class PlayerController : MonoBehaviour
{
    [Header("References")]
    public Transform playerCamera;
    [Tooltip("Used to spawn on the ground and to recover if you ever fall through.")]
    public NeuralTerrain terrain;

    [Header("Movement")]
    public float walkSpeed = 6f;
    public float sprintSpeed = 14f;
    [Tooltip("How fast you reach full speed / stop. Higher = snappier.")]
    public float acceleration = 10f;
    public float jumpHeight = 1.3f;
    public float gravity = -22f;

    [Header("Water")]
    [Tooltip("Water deeper than this (from the feet) makes you swim, keeping your head above the surface.")]
    public float swimDepth = 1.3f;
    public float swimSpeed = 3f;

    [Header("Look")]
    [Tooltip("Degrees per mouse unit. Frame-rate independent.")]
    public float mouseSensitivity = 2f;
    public float eyeHeight = 1.7f;
    [Tooltip("Smooths the camera when walking up and down steps/bumps (0 = off).")]
    [Range(0f, 0.2f)] public float stepSmoothing = 0.08f;

    CharacterController _cc;
    Vector3 _velocity;          // horizontal velocity in x/z, vertical in y
    float _pitch, _yaw;
    float _smoothedEyeY;

    void Start()
    {
        _cc = GetComponent<CharacterController>();
        if (terrain == null) terrain = FindAnyObjectByType<NeuralTerrain>(); // prefab dropped into a new scene
        if (playerCamera == null && GetComponentInChildren<Camera>() is Camera cam) playerCamera = cam.transform;
        _yaw = transform.eulerAngles.y;
        if (playerCamera != null) playerCamera.localPosition = new Vector3(0f, eyeHeight - _cc.height * 0.5f, 0f);
        SnapToGround();
        _smoothedEyeY = transform.position.y;
        LockCursor(true);
    }

    void Update()
    {
        if (Input.GetKeyDown(KeyCode.Escape)) LockCursor(false);
        else if (Input.GetMouseButtonDown(0) && Cursor.lockState != CursorLockMode.Locked) LockCursor(true);

        if (Cursor.lockState == CursorLockMode.Locked) Look();
        Move();

        // Fell through the world (e.g. terrain regenerated under you) -> put back on top.
        if (terrain != null && transform.position.y < terrain.SampleHeight(transform.position.x, transform.position.z) - 5f)
            SnapToGround();
    }

    void LateUpdate()
    {
        // Camera follows body height with a tiny lag so steps feel smooth, not jittery.
        if (playerCamera == null) return;
        float t = stepSmoothing <= 0f ? 1f : 1f - Mathf.Exp(-Time.deltaTime / stepSmoothing);
        _smoothedEyeY = Mathf.Lerp(_smoothedEyeY, transform.position.y, t);
        playerCamera.localPosition = new Vector3(0f, eyeHeight - _cc.height * 0.5f + (_smoothedEyeY - transform.position.y), 0f);
    }

    void Look()
    {
        // Mouse axes are already "movement this frame" -- no deltaTime here.
        _yaw += Input.GetAxisRaw("Mouse X") * mouseSensitivity;
        _pitch = Mathf.Clamp(_pitch - Input.GetAxisRaw("Mouse Y") * mouseSensitivity, -85f, 85f);
        transform.rotation = Quaternion.Euler(0f, _yaw, 0f);
        if (playerCamera != null) playerCamera.localRotation = Quaternion.Euler(_pitch, 0f, 0f);
    }

    void Move()
    {
        Vector2 input = new Vector2(Input.GetAxisRaw("Horizontal"), Input.GetAxisRaw("Vertical"));
        input = Vector2.ClampMagnitude(input, 1f);

        // Deep water: float with the head above the surface instead of walking on the river bed.
        Vector3 pos = transform.position;
        float feet = pos.y - _cc.height * 0.5f;
        float surface = terrain != null ? terrain.SampleWater(pos.x, pos.z) : float.NegativeInfinity;
        bool swimming = surface - feet > swimDepth;

        float speed = swimming ? swimSpeed : Input.GetKey(KeyCode.LeftShift) ? sprintSpeed : walkSpeed;
        Vector3 target = (transform.right * input.x + transform.forward * input.y) * speed;

        // Smooth acceleration on the ground, less control in the air.
        float control = _cc.isGrounded || swimming ? 1f : 0.3f;
        float k = 1f - Mathf.Exp(-acceleration * control * Time.deltaTime);
        _velocity.x = Mathf.Lerp(_velocity.x, target.x, k);
        _velocity.z = Mathf.Lerp(_velocity.z, target.z, k);

        if (swimming)
        {
            // Spring toward floating height; Space bobs up so you can climb out onto a bank.
            float floatFeet = surface - swimDepth + 0.1f;
            _velocity.y = Mathf.Clamp((floatFeet - feet) * 4f, -4f, 4f);
            if (Input.GetButton("Jump")) _velocity.y = 3f;
        }
        else if (_cc.isGrounded)
        {
            _velocity.y = -2f; // keeps us glued to slopes when walking downhill
            if (Input.GetButtonDown("Jump")) _velocity.y = Mathf.Sqrt(jumpHeight * -2f * gravity);
        }
        if (!swimming) _velocity.y += gravity * Time.deltaTime;

        _cc.Move(_velocity * Time.deltaTime);
    }

    void SnapToGround()
    {
        if (terrain == null) return;
        Vector3 p = transform.position;
        p.y = terrain.SampleHeight(p.x, p.z) + _cc.height * 0.5f + _cc.skinWidth + 0.2f;
        _cc.enabled = false; // CharacterController ignores direct position changes while enabled
        transform.position = p;
        _cc.enabled = true;
        _velocity = Vector3.zero;
    }

    static void LockCursor(bool locked)
    {
        Cursor.lockState = locked ? CursorLockMode.Locked : CursorLockMode.None;
        Cursor.visible = !locked;
    }
}
