#!/usr/bin/env python3
"""
PyChrono Demo: 2D Robotic Arm Simulation
=========================================

This demo simulates a simple 2-link robotic arm in a plane using PyChrono.
It demonstrates:
- Creating rigid bodies (links, base)
- Revolute joints between links
- Motor actuation
- Visualization with Irrlicht

Requirements:
- pychrono 9.0.1

"""

import math
import pychrono.core as chrono
import pychrono.irrlicht as irr

# =============================================================================
# SIMULATION PARAMETERS
# =============================================================================

# Timestep
TIME_STEP = 0.01

# Simulation duration
SIM_DURATION = 10.0

# Gravity
GRAVITY = chrono.ChVector3d(0, -9.81, 0)

# =============================================================================
# GEOMETRY PARAMETERS
# =============================================================================

# Base parameters
BASE_SIZE = chrono.ChVector3d(0.3, 0.1, 0.3)
BASE_MASS = 10.0

# Link parameters
LINK_LENGTH = 1.0
LINK_RADIUS = 0.05
LINK_MASS = 1.0

# Unit conversions
DEG_TO_RAD = math.pi / 180.0
RAD_TO_DEG = 180.0 / math.pi


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def q_align_localZ_to_dir(dx, dy):
    """Return a ChQuaterniond that orients local +Z to point along (dx,dy) in XY plane."""
    L = math.hypot(dx, dy)
    if L == 0.0:
        return chrono.ChQuaterniond(1.0, 0.0, 0.0, 0.0)
    dx_hat = dx / L
    dy_hat = dy / L
    ax_x = -dy_hat
    ax_y = dx_hat
    ax_z = 0.0
    axis_len = math.hypot(ax_x, ax_y)
    if axis_len == 0.0:
        return chrono.ChQuaterniond(1.0, 0.0, 0.0, 0.0)
    ax_x /= axis_len
    ax_y /= axis_len
    # rotate 90 degrees about axis (ax_x,ax_y,0) using quaternion formula
    half = 0.5 * math.pi
    s = math.sin(half)
    return chrono.ChQuaterniond(math.cos(half), ax_x*s, ax_y*s, ax_z*s)


def create_cylinder_body(system, radius, length, mass, pos, rot):
    """
    Create a cylindrical rigid body and add it to the system.
    
    Parameters:
    -----------
    system : ChSystem
        The physical system
    radius : float
        Cylinder radius
    length : float
        Cylinder length (along Z axis)
    mass : float
        Body mass
    pos : ChVector3d
        Position of body center
    rot : ChQuaterniond
        Orientation quaternion
        
    Returns:
    --------
    ChBody : The created body
    """
    body = chrono.ChBody()
    body.SetPos(pos)
    body.SetRot(rot)
    body.SetMass(mass)
    
    # Calculate inertia for cylinder
    Ixx = (1.0/12.0) * mass * (3*radius*radius + length*length)
    Iyy = Ixx
    Izz = 0.5 * mass * radius * radius
    body.SetInertiaXX(chrono.ChVector3d(Ixx, Iyy, Izz))
    
    # Add visualization
    cyl_shape = chrono.ChVisualShapeCylinder(radius, length)
    cyl_shape.SetColor(chrono.ChColor(0.6, 0.6, 0.8))
    body.AddVisualShape(cyl_shape, chrono.ChFramed())
    
    # Add collision (optional, not used in this demo)
    # body.EnableCollision(False)
    
    system.Add(body)
    return body


def create_box_body(system, size, mass, pos, rot):
    """
    Create a box-shaped rigid body.
    
    Parameters:
    -----------
    system : ChSystem
        The physical system
    size : ChVector3d
        Box dimensions (x, y, z)
    mass : float
        Body mass
    pos : ChVector3d
        Position of body center
    rot : ChQuaterniond
        Orientation quaternion
        
    Returns:
    --------
    ChBody : The created body
    """
    body = chrono.ChBody()
    body.SetPos(pos)
    body.SetRot(rot)
    body.SetMass(mass)
    body.SetFixed(True)  # Base is fixed
    
    # Calculate inertia for box
    Ixx = (1.0/12.0) * mass * (size.y*size.y + size.z*size.z)
    Iyy = (1.0/12.0) * mass * (size.x*size.x + size.z*size.z)
    Izz = (1.0/12.0) * mass * (size.x*size.x + size.y*size.y)
    body.SetInertiaXX(chrono.ChVector3d(Ixx, Iyy, Izz))
    
    # Add visualization
    box_shape = chrono.ChVisualShapeBox(size.x, size.y, size.z)
    box_shape.SetColor(chrono.ChColor(0.8, 0.6, 0.4))
    body.AddVisualShape(box_shape, chrono.ChFramed())
    
    system.Add(body)
    return body


def make_arm(system, base_body, link1_length, link1_radius, link1_mass,
             link2_length, link2_radius, link2_mass):
    """
    Create a two-link arm attached to a base body.
    
    Parameters:
    -----------
    system : ChSystem
        The physical system
    base_body : ChBody
        The base body to attach the arm to
    link1_length : float
        Length of first link
    link1_radius : float
        Radius of first link
    link1_mass : float
        Mass of first link
    link2_length : float
        Length of second link
    link2_radius : float
        Radius of second link
    link2_mass : float
        Mass of second link
        
    Returns:
    --------
    tuple : (link1_body, link2_body, joint1, joint2, motor)
    """
    
    # Get base position
    base_pos = base_body.GetPos()
    
    # Create Link 1
    # Position it so its base is at the top of the base_body
    link1_pos = chrono.ChVector3d(
        base_pos.x,
        base_pos.y + BASE_SIZE.y/2 + link1_length/2,
        base_pos.z
    )
    
    # Orientation: initially vertical (along +Y)
    # Use helper to align local Z to direction
    link1_rot = q_align_localZ_to_dir(0.0, 1.0)
    
    link1 = create_cylinder_body(
        system, link1_radius, link1_length, link1_mass,
        link1_pos, link1_rot
    )
    link1.SetFixed(False)
    
    # Create revolute joint between base and link1
    # Joint axis is Z (perpendicular to XY plane)
    joint1_frame_base = chrono.ChFramed(
        chrono.ChVector3d(base_pos.x, base_pos.y + BASE_SIZE.y/2, base_pos.z),
        chrono.ChQuaterniond(1, 0, 0, 0)
    )
    joint1_frame_link1 = chrono.ChFramed(
        chrono.ChVector3d(link1_pos.x, link1_pos.y - link1_length/2, link1_pos.z),
        chrono.ChQuaterniond(1, 0, 0, 0)
    )
    
    joint1 = chrono.ChLinkRevolute()
    joint1.Initialize(base_body, link1, joint1_frame_base, joint1_frame_link1)
    system.Add(joint1)
    
    # Create Link 2
    # Position it at the end of Link 1
    link2_pos = chrono.ChVector3d(
        link1_pos.x,
        link1_pos.y + link1_length/2 + link2_length/2,
        link1_pos.z
    )
    
    link2_rot = q_align_localZ_to_dir(0.0, 1.0)
    
    link2 = create_cylinder_body(
        system, link2_radius, link2_length, link2_mass,
        link2_pos, link2_rot
    )
    link2.SetFixed(False)
    
    # Create revolute joint between link1 and link2
    joint2_frame_link1 = chrono.ChFramed(
        chrono.ChVector3d(link1_pos.x, link1_pos.y + link1_length/2, link1_pos.z),
        chrono.ChQuaterniond(1, 0, 0, 0)
    )
    joint2_frame_link2 = chrono.ChFramed(
        chrono.ChVector3d(link2_pos.x, link2_pos.y - link2_length/2, link2_pos.z),
        chrono.ChQuaterniond(1, 0, 0, 0)
    )
    
    joint2 = chrono.ChLinkRevolute()
    joint2.Initialize(link1, link2, joint2_frame_link1, joint2_frame_link2)
    system.Add(joint2)
    
    # Add a motor to joint1 for actuation
    motor = chrono.ChLinkMotorRotationSpeed()
    motor.Initialize(base_body, link1, joint1_frame_base)
    
    # Set motor speed function (sinusoidal motion)
    speed_func = chrono.ChFunctionRamp(0, 0.5)  # Ramp function: m*t + q
    motor.SetSpeedFunction(speed_func)
    system.Add(motor)
    
    return link1, link2, joint1, joint2, motor


# =============================================================================
# MAIN SIMULATION FUNCTION
# =============================================================================

def main():
    """
    Main simulation function.
    Sets up the physical system, creates bodies, and runs the simulation.
    """
    
    print("="*70)
    print("PyChrono 2-Link Robotic Arm Demo")
    print("="*70)
    
    # Create the physical system
    system = chrono.ChSystemNSC()
    system.SetGravitationalAcceleration(GRAVITY)
    
    # Create the ground/base platform
    base_pos = chrono.ChVector3d(0, 0, 0)
    base_rot = chrono.ChQuaterniond(1, 0, 0, 0)
    
    base = create_box_body(
        system, BASE_SIZE, BASE_MASS,
        base_pos, base_rot
    )
    
    print(f"Created base at {base_pos}")
    
    # Create the robotic arm
    link1, link2, joint1, joint2, motor = make_arm(
        system, base,
        LINK_LENGTH, LINK_RADIUS, LINK_MASS,
        LINK_LENGTH, LINK_RADIUS, LINK_MASS
    )
    
    print(f"Created 2-link arm")
    print(f"  Link 1: length={LINK_LENGTH}, mass={LINK_MASS}")
    print(f"  Link 2: length={LINK_LENGTH}, mass={LINK_MASS}")
    
    # Create visualization system
    vis = irr.ChVisualSystemIrrlicht()
    vis.AttachSystem(system)
    vis.SetWindowSize(1024, 768)
    vis.SetWindowTitle("PyChrono Robotic Arm Demo")
    vis.Initialize()
    vis.AddCamera(chrono.ChVector3d(3, 2, 3), chrono.ChVector3d(0, 1, 0))
    vis.AddTypicalLights()
    
    print("Visualization initialized")
    
    # Simulation loop
    print(f"Starting simulation (duration={SIM_DURATION}s, dt={TIME_STEP}s)")
    
    time = 0.0
    step_count = 0
    
    while vis.Run() and time < SIM_DURATION:
        # Update visualization
        vis.BeginScene()
        vis.Render()
        vis.EndScene()
        
        # Advance simulation
        system.DoStepDynamics(TIME_STEP)
        
        time += TIME_STEP
        step_count += 1
        
        # Print status every second
        if step_count % int(1.0/TIME_STEP) == 0:
            link1_pos = link1.GetPos()
            link2_pos = link2.GetPos()
            print(f"t={time:.2f}s: Link1 Y={link1_pos.y:.3f}, Link2 Y={link2_pos.y:.3f}")
    
    print("Simulation complete")
    print(f"  Total steps: {step_count}")
    print(f"  Final time: {time:.2f}s")


# =============================================================================
# ADDITIONAL UTILITY FUNCTIONS
# =============================================================================

def print_system_info(system):
    """Print information about the system state."""
    print("\n" + "="*70)
    print("SYSTEM INFORMATION")
    print("="*70)
    print(f"Number of bodies: {system.GetNumBodies()}")
    print(f"Number of links: {system.GetNumLinks()}")
    print(f"System time: {system.GetChTime():.4f}s")
    
    # Print body information
    print("\nBodies:")
    for i in range(system.GetNumBodies()):
        body = system.GetBodies()[i]
        pos = body.GetPos()
        vel = body.GetPosDt()
        print(f"  Body {i}: pos=({pos.x:.3f}, {pos.y:.3f}, {pos.z:.3f}), "
              f"vel=({vel.x:.3f}, {vel.y:.3f}, {vel.z:.3f})")


def compute_end_effector_position(base_pos, link1_length, link2_length, 
                                   theta1, theta2):
    """
    Compute end effector position given joint angles.
    
    Parameters:
    -----------
    base_pos : ChVector3d
        Base position
    link1_length : float
        Length of first link
    link2_length : float
        Length of second link
    theta1 : float
        Angle of first joint (radians)
    theta2 : float
        Angle of second joint (radians)
        
    Returns:
    --------
    ChVector3d : End effector position
    """
    # Forward kinematics
    x = link1_length * math.cos(theta1) + link2_length * math.cos(theta1 + theta2)
    y = link1_length * math.sin(theta1) + link2_length * math.sin(theta1 + theta2)
    
    return chrono.ChVector3d(
        base_pos.x + x,
        base_pos.y + y,
        base_pos.z
    )


def inverse_kinematics(target_x, target_y, link1_length, link2_length):
    """
    Compute joint angles to reach target position (2D).
    
    Parameters:
    -----------
    target_x : float
        Target X coordinate
    target_y : float
        Target Y coordinate
    link1_length : float
        Length of first link
    link2_length : float
        Length of second link
        
    Returns:
    --------
    tuple : (theta1, theta2) in radians, or None if unreachable
    """
    # Distance to target
    d = math.hypot(target_x, target_y)
    
    # Check if reachable
    if d > link1_length + link2_length or d < abs(link1_length - link2_length):
        return None
    
    # Cosine law for theta2
    cos_theta2 = (d*d - link1_length*link1_length - link2_length*link2_length) / (2 * link1_length * link2_length)
    cos_theta2 = max(-1.0, min(1.0, cos_theta2))  # Clamp to [-1, 1]
    theta2 = math.acos(cos_theta2)
    
    # Calculate theta1
    k1 = link1_length + link2_length * math.cos(theta2)
    k2 = link2_length * math.sin(theta2)
    theta1 = math.atan2(target_y, target_x) - math.atan2(k2, k1)
    
    return theta1, theta2


def create_trajectory_points(center, radius, num_points):
    """
    Generate circular trajectory points.
    
    Parameters:
    -----------
    center : tuple
        (x, y) center of circle
    radius : float
        Circle radius
    num_points : int
        Number of points on circle
        
    Returns:
    --------
    list : List of (x, y) tuples
    """
    points = []
    for i in range(num_points):
        angle = 2 * math.pi * i / num_points
        x = center[0] + radius * math.cos(angle)
        y = center[1] + radius * math.sin(angle)
        points.append((x, y))
    return points


def compute_jacobian(theta1, theta2, link1_length, link2_length):
    """
    Compute the Jacobian matrix for the 2-link arm.
    
    Parameters:
    -----------
    theta1 : float
        Angle of first joint (radians)
    theta2 : float
        Angle of second joint (radians)
    link1_length : float
        Length of first link
    link2_length : float
        Length of second link
        
    Returns:
    --------
    list : 2x2 Jacobian matrix as list of lists
    """
    c1 = math.cos(theta1)
    s1 = math.sin(theta1)
    c12 = math.cos(theta1 + theta2)
    s12 = math.sin(theta1 + theta2)
    
    j11 = -link1_length * s1 - link2_length * s12
    j12 = -link2_length * s12
    j21 = link1_length * c1 + link2_length * c12
    j22 = link2_length * c12
    
    return [[j11, j12], [j21, j22]]


def compute_workspace_limits(link1_length, link2_length):
    """
    Compute the workspace limits for the 2-link arm.
    
    Parameters:
    -----------
    link1_length : float
        Length of first link
    link2_length : float
        Length of second link
        
    Returns:
    --------
    dict : Dictionary with 'min_radius' and 'max_radius'
    """
    min_radius = abs(link1_length - link2_length)
    max_radius = link1_length + link2_length
    return {'min_radius': min_radius, 'max_radius': max_radius}


# =============================================================================
# PLOTTING AND ANALYSIS FUNCTIONS
# =============================================================================

def plot_arm_configuration(theta1, theta2, link1_length, link2_length):
    """
    Plot the arm configuration (requires matplotlib).
    
    This function would create a visualization of the arm position.
    Note: matplotlib is not required for the main simulation.
    
    Parameters:
    -----------
    theta1 : float
        Angle of first joint (radians)
    theta2 : float
        Angle of second joint (radians)
    link1_length : float
        Length of first link
    link2_length : float
        Length of second link
    """
    # Note: This is a placeholder. Actual implementation would use matplotlib.
    # For the demo, we'll just print the configuration.
    print(f"\nArm Configuration:")
    print(f"  Joint 1 angle: {theta1 * RAD_TO_DEG:.2f} degrees")
    print(f"  Joint 2 angle: {theta2 * RAD_TO_DEG:.2f} degrees")
    
    # Compute positions
    x1 = link1_length * math.cos(theta1)
    y1 = link1_length * math.sin(theta1)
    x2 = x1 + link2_length * math.cos(theta1 + theta2)
    y2 = y1 + link2_length * math.sin(theta1 + theta2)
    
    print(f"  Joint 1 position: ({x1:.3f}, {y1:.3f})")
    print(f"  End effector position: ({x2:.3f}, {y2:.3f})")


def analyze_dynamics(system, duration, dt):
    """
    Analyze system dynamics over time.
    
    Parameters:
    -----------
    system : ChSystem
        The physical system
    duration : float
        Analysis duration
    dt : float
        Time step
        
    Returns:
    --------
    dict : Dictionary with time series data
    """
    times = []
    energies = []
    
    time = 0.0
    while time < duration:
        system.DoStepDynamics(dt)
        time += dt
        
        # Compute total energy
        ke = 0.0
        pe = 0.0
        for body in system.GetBodies():
            if not body.IsFixed():
                vel = body.GetPosDt()
                mass = body.GetMass()
                ke += 0.5 * mass * vel.Length2()
                
                pos = body.GetPos()
                pe += mass * abs(GRAVITY.y) * pos.y
        
        times.append(time)
        energies.append(ke + pe)
    
    return {'time': times, 'energy': energies}


def export_trajectory(filename, positions, times):
    """
    Export trajectory data to a file.
    
    Parameters:
    -----------
    filename : str
        Output filename
    positions : list
        List of ChVector3d positions
    times : list
        List of corresponding times
    """
    with open(filename, 'w') as f:
        f.write("time,x,y,z\n")
        for t, pos in zip(times, positions):
            f.write(f"{t:.4f},{pos.x:.6f},{pos.y:.6f},{pos.z:.6f}\n")
    print(f"Trajectory exported to {filename}")


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nSimulation interrupted by user")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
