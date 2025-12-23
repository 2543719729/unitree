import torch


def joint_mirror_math(q: torch.Tensor, mirror_pairs: list[tuple[int, int]]) -> torch.Tensor:
    pen = torch.zeros(q.shape[0], device=q.device)
    for li, ri in mirror_pairs:
        pen = pen + torch.square(q[:, li] - q[:, ri])
    if len(mirror_pairs) > 0:
        pen = pen * (1.0 / len(mirror_pairs))
    return pen


def joint_symmetry_out_of_phase_math(
    q: torch.Tensor,
    q0: torch.Tensor,
    pitch_pairs: list[tuple[int, int]],
    roll_pairs: list[tuple[int, int]] | None = None,
    yaw_pairs: list[tuple[int, int]] | None = None,
    left_contact: torch.Tensor | None = None,
    right_contact: torch.Tensor | None = None,
    roll_scale: float = 0.2,
    yaw_scale: float = 0.1,
) -> torch.Tensor:
    if roll_pairs is None:
        roll_pairs = []
    if yaw_pairs is None:
        yaw_pairs = []

    if left_contact is None or right_contact is None:
        w_in = torch.zeros(q.shape[0], device=q.device)
        w_out = torch.ones(q.shape[0], device=q.device)
    else:
        w_in = (left_contact & right_contact).float()
        w_out = (left_contact ^ right_contact).float()

    w_active = w_in + w_out

    pitch_pen = torch.zeros(q.shape[0], device=q.device)
    for li, ri in pitch_pairs:
        l_rel = q[:, li] - q0[:, li]
        r_rel = q[:, ri] - q0[:, ri]
        pitch_pen = pitch_pen + w_out * torch.square(l_rel + r_rel) + w_in * torch.square(l_rel - r_rel)
    if len(pitch_pairs) > 0:
        pitch_pen = pitch_pen * (1.0 / len(pitch_pairs))

    roll_pen = torch.zeros(q.shape[0], device=q.device)
    for li, ri in roll_pairs:
        l_rel = q[:, li] - q0[:, li]
        r_rel = q[:, ri] - q0[:, ri]
        roll_pen = roll_pen + w_active * torch.square(l_rel - r_rel)
    if len(roll_pairs) > 0:
        roll_pen = roll_pen * (1.0 / len(roll_pairs))

    yaw_pen = torch.zeros(q.shape[0], device=q.device)
    for li, ri in yaw_pairs:
        l_rel = q[:, li] - q0[:, li]
        r_rel = q[:, ri] - q0[:, ri]
        yaw_pen = yaw_pen + w_active * torch.square(l_rel - r_rel)
    if len(yaw_pairs) > 0:
        yaw_pen = yaw_pen * (1.0 / len(yaw_pairs))

    return pitch_pen + roll_scale * roll_pen + yaw_scale * yaw_pen


def main() -> None:
    device = "cpu"
    num_envs = 1

    joint_index = {
        "l_hip_pitch": 0,
        "r_hip_pitch": 1,
        "l_knee": 2,
        "r_knee": 3,
        "l_ankle_pitch": 4,
        "r_ankle_pitch": 5,
        "l_hip_roll": 6,
        "r_hip_roll": 7,
        "l_hip_yaw": 8,
        "r_hip_yaw": 9,
    }

    pitch_pairs = [
        (joint_index["l_hip_pitch"], joint_index["r_hip_pitch"]),
        (joint_index["l_knee"], joint_index["r_knee"]),
        (joint_index["l_ankle_pitch"], joint_index["r_ankle_pitch"]),
    ]
    roll_pairs = [(joint_index["l_hip_roll"], joint_index["r_hip_roll"])]
    yaw_pairs = [(joint_index["l_hip_yaw"], joint_index["r_hip_yaw"])]

    mirror_pairs = pitch_pairs + roll_pairs + yaw_pairs

    q0 = torch.zeros(num_envs, len(joint_index), device=device)

    # Case A: "walking-like" (out-of-phase) pose
    # left forward, right backward
    q_a = q0.clone()
    q_a[:, joint_index["l_hip_pitch"]] = 0.3
    q_a[:, joint_index["r_hip_pitch"]] = -0.3

    # Case B: "bunny-hop-like" (in-phase) pose
    # both forward
    q_b = q0.clone()
    q_b[:, joint_index["l_hip_pitch"]] = 0.3
    q_b[:, joint_index["r_hip_pitch"]] = 0.3

    left_contact = torch.tensor([True], device=device)
    right_contact = torch.tensor([False], device=device)

    print("=== Single support (expect out-of-phase symmetry for pitch) ===")
    print("joint_mirror   A(out-of-phase):", float(joint_mirror_math(q_a, mirror_pairs)[0]))
    print("joint_mirror   B(in-phase):   ", float(joint_mirror_math(q_b, mirror_pairs)[0]))
    print("new_symmetry   A(out-of-phase):", float(joint_symmetry_out_of_phase_math(q_a, q0, pitch_pairs, roll_pairs, yaw_pairs, left_contact, right_contact)[0]))
    print("new_symmetry   B(in-phase):   ", float(joint_symmetry_out_of_phase_math(q_b, q0, pitch_pairs, roll_pairs, yaw_pairs, left_contact, right_contact)[0]))

    left_contact_ds = torch.tensor([True], device=device)
    right_contact_ds = torch.tensor([True], device=device)

    print("\n=== Double support (allow in-phase, avoid forcing anti-phase) ===")
    print("new_symmetry   A(out-of-phase):", float(joint_symmetry_out_of_phase_math(q_a, q0, pitch_pairs, roll_pairs, yaw_pairs, left_contact_ds, right_contact_ds)[0]))
    print("new_symmetry   B(in-phase):   ", float(joint_symmetry_out_of_phase_math(q_b, q0, pitch_pairs, roll_pairs, yaw_pairs, left_contact_ds, right_contact_ds)[0]))

    left_contact_f = torch.tensor([False], device=device)
    right_contact_f = torch.tensor([False], device=device)

    print("\n=== Flight / no-contact (symmetry disabled) ===")
    print("new_symmetry   A(out-of-phase):", float(joint_symmetry_out_of_phase_math(q_a, q0, pitch_pairs, roll_pairs, yaw_pairs, left_contact_f, right_contact_f)[0]))
    print("new_symmetry   B(in-phase):   ", float(joint_symmetry_out_of_phase_math(q_b, q0, pitch_pairs, roll_pairs, yaw_pairs, left_contact_f, right_contact_f)[0]))


if __name__ == "__main__":
    main()
