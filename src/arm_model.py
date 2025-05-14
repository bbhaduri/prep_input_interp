import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

# Define physical properties of arm model
# arm lengths
L1, L2 = 0.30, 0.30 #m
# arm masses
M1, M2 = 1.4, 1.0 #kg
# moments of inertia
I1, I2 = .025, .045 #kg/m^2
# center of mass for lower arm
D2 = .16 #m

# constants for dynamics
a1 = I1 + I2 + (M2*L1**2)
a2 = (M2*L1*D2)
a3 = I2

# Constant damping matrix
B = jnp.array(
    [(.05, .025),
    (.025, .05)]
)

# Add Theta Constraints
max_theta = jnp.pi
min_theta = 0.

# Define delta t
dt = 0.001

# Define functions to calculate matrices for forward dynamics
def calc_dyn_mats(theta1, theta2, dtheta1, dtheta2):
    """
    """
    # Update Matrix of inertia
    m_theta = jnp.array(
        [(a1+2*a2*jnp.cos(theta2), a3+a2*jnp.cos(theta2)),
        (a3+a2*jnp.cos(theta2), a3)]
    )
    # Update Centripetal and Coriolis forces
    Cor = (a2*jnp.sin(theta2)) * jnp.array(
        [(-dtheta2*(2*dtheta1+dtheta2)), (dtheta1**2)]
    )

    return m_theta, Cor

# Define function for calculating arm positions
def calc_arm_pos(thetas):
    """
    """
    # Extract necessary state vars
    theta1, theta2 = thetas
    # Calculate positions and return
    elbow_pos = jnp.array([
        (L1*jnp.cos(theta1)), (L1*jnp.sin(theta1))
    ])

    hand_pos = jnp.array(
        [(elbow_pos[0] + L2*jnp.cos(theta1+theta2)),
         (elbow_pos[1] + L2*jnp.sin(theta1+theta2))]
    )

    return jnp.vstack([hand_pos, elbow_pos])

def get_angles_from_pos(pos):
    """
    """
    x, y = pos
    theta1 = jnp.arctan2(y, x) - jnp.arccos((x**2 + y**2 + L1**2 - L2**2)
                                                     /(2*L1*(x**2 + y**2)**0.5))
    theta2 = jnp.arccos((x**2 + y**2 - L1**2 - L2**2)/(2*L1*L2))

    return theta1, theta2

def init_radial_task(start_pos=jnp.array([0.0, 0.4]), radius=0.12):
    """
    """
    # Set starting positions
    x0, y0 = start_pos

    # Get target locations
    target_angles = jnp.array([0., 45., 90., 135., 180., 225., 270., 315.])*(2*jnp.pi/360)
    target_x = x0 + (jnp.cos(target_angles)*radius)
    target_y = y0 + (jnp.sin(target_angles)*radius)
    targets = jnp.concat([target_x[:, None], target_y[:, None]], axis=1) #m

    # Get initial angles from starting position
    theta1, theta2 = get_angles_from_pos(start_pos)
    init_thetas = jnp.vstack([theta1, theta2])

    # Store arm angles for targets
    target_t1, target_t2 = jax.vmap(get_angles_from_pos)(targets)
    target_angles = jnp.hstack([target_t1[:, None], target_t2[:, None]])
    
    return init_thetas, target_angles, targets

def check_bounds(n_state):
    """
    Check if theta1 and theta2 are out of biomechanical-ish bounds.
    """

    # Check thetas against upper bound
    n_state = n_state.at[0].set(jax.lax.select(n_state[0] > max_theta, max_theta, n_state[0]))
    n_state = n_state.at[1].set(jax.lax.select(n_state[1] > max_theta, max_theta, n_state[1]))

    # Check thetas against lower bound
    n_state = n_state.at[0].set(jax.lax.select(n_state[0] < min_theta, min_theta, n_state[0]))
    n_state = n_state.at[1].set(jax.lax.select(n_state[1] < min_theta, min_theta, n_state[1]))

    # Set angular velocities to 0 if bounds are reached
    n_state = n_state.at[2].set(
        jax.lax.select(
            jnp.logical_or(n_state[0] == min_theta, n_state[0] == max_theta),
            0.,
            n_state[2]
        )
    )
    n_state = n_state.at[3].set(
        jax.lax.select(
            jnp.logical_or(n_state[1] == min_theta, n_state[1] == max_theta),
            0.,
            n_state[3]
        )
    )

    return n_state

# Define dynamics step
def update_state(state, torques):
    """
    """
    arm_state = state[-4:]
    # extract state vars
    theta1, theta2, dtheta1, dtheta2 = arm_state
    
    # Get only angular velocities
    dthetas = arm_state[2:]
    # Update dynamics matrices
    m_theta, Cor = calc_dyn_mats(
        theta1, theta2, dtheta1, dtheta2
    )
    # Forward dynamics of torques applied to arm
    d2thetas = jnp.linalg.inv(m_theta) @ (torques - Cor + (B@dthetas))
    
    # New state
    dstate = jnp.vstack([dthetas[:, None], d2thetas[:, None]]).squeeze()

    # Update state (TODO: May want to use a more powerful integration method)
    n_state = arm_state + dt*dstate

    # Check Bounds and return new state
    return check_bounds(n_state)

