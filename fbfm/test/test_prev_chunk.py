import unittest
import torch
import sys
import os

# Add the fbfm directory to the path so we can import the modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fbfm.policies.fbfm.modeling_rtc_fbfm import PrevChunk

class TestPrevChunk(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.action_num = 16
        self.action_dim = 16
        self.state_num = 4
        self.state_dim = 128
        
        # Create a PrevChunk instance for testing
        self.prev_chunk = PrevChunk(
            action_num=self.action_num,
            action_dim=self.action_dim,
            state_num=self.state_num,
            state_dim=self.state_dim
        )
    
    def test_initialization(self):
        """Test that PrevChunk initializes correctly with default values."""
        self.assertEqual(self.prev_chunk.constrain_mode, "Feedback")
        self.assertEqual(self.prev_chunk.action_constrained_num, 0)
        self.assertEqual(self.prev_chunk.state_constrained_num, 0)
        self.assertEqual(self.prev_chunk.inference_delay, 0)
        self.assertEqual(self.prev_chunk.action_num, self.action_num)
        self.assertEqual(self.prev_chunk.action_dim, self.action_dim)
        self.assertEqual(self.prev_chunk.state_num, self.state_num)
        self.assertEqual(self.prev_chunk.state_dim, self.state_dim)
        
        # Check that actions and states are initialized as zero tensors with fixed shape
        self.assertIsNotNone(self.prev_chunk.actions)
        self.assertEqual(self.prev_chunk.actions.shape, (self.action_num, self.action_dim))
        self.assertTrue(torch.all(self.prev_chunk.actions == 0))
        
        self.assertIsNotNone(self.prev_chunk.states)
        self.assertEqual(self.prev_chunk.states.shape, (self.state_num, self.state_dim))
        self.assertTrue(torch.all(self.prev_chunk.states == 0))
    
    def test_append_new_state_first_observation(self):
        """Test appending the first state observation."""
        # Create a new state tensor
        new_state = torch.randn(1, self.state_dim)
        
        # Append the new state
        self.prev_chunk.append_new_state(new_state)
        
        # Check that states has fixed shape and contains the new state at index 0
        self.assertEqual(self.prev_chunk.states.shape, (self.state_num, self.state_dim))
        self.assertTrue(torch.equal(self.prev_chunk.states[0], new_state[0]))
        
        # Check that remaining states are still zero
        self.assertTrue(torch.all(self.prev_chunk.states[1:] == 0))
        
        # Check that state_constrained_num is updated
        self.assertEqual(self.prev_chunk.state_constrained_num, 1)
    
    def test_append_new_state_incremental(self):
        """Test appending multiple state observations incrementally."""
        # Append first state
        first_state = torch.randn(1, self.state_dim)
        self.prev_chunk.append_new_state(first_state)
        
        # Append second state
        second_state = torch.randn(1, self.state_dim)
        self.prev_chunk.append_new_state(second_state)
        
        # Check that both states are stored at correct positions
        self.assertEqual(self.prev_chunk.states.shape, (self.state_num, self.state_dim))
        self.assertTrue(torch.equal(self.prev_chunk.states[0], first_state[0]))
        self.assertTrue(torch.equal(self.prev_chunk.states[1], second_state[0]))
        
        # Check that remaining states are still zero
        self.assertTrue(torch.all(self.prev_chunk.states[2:] == 0))
        
        # Check that state_constrained_num is updated
        self.assertEqual(self.prev_chunk.state_constrained_num, 2)
    
    def test_append_new_state_1d_tensor(self):
        """Test appending a 1D tensor (should be automatically converted to 2D)."""
        # Create a 1D state tensor
        new_state_1d = torch.randn(self.state_dim)
        
        # Append the 1D state
        self.prev_chunk.append_new_state(new_state_1d)
        
        # Check that it was properly converted and stored at index 0
        self.assertEqual(self.prev_chunk.states.shape, (self.state_num, self.state_dim))
        self.assertTrue(torch.equal(self.prev_chunk.states[0], new_state_1d))
        
        # Check that remaining states are still zero
        self.assertTrue(torch.all(self.prev_chunk.states[1:] == 0))
    
    def test_get_constrained_states(self):
        """Test getting constrained states - should always return fixed shape tensor."""
        # Should return zero tensor with fixed shape even when no states added
        constrained_states = self.prev_chunk.get_constrained_states()
        self.assertEqual(constrained_states.shape, (self.state_num, self.state_dim))
        self.assertTrue(torch.all(constrained_states == 0))
        
        # Add some states and test again
        new_state = torch.randn(1, self.state_dim)
        self.prev_chunk.append_new_state(new_state)
        constrained_states = self.prev_chunk.get_constrained_states()
        self.assertEqual(constrained_states.shape, (self.state_num, self.state_dim))
        self.assertTrue(torch.equal(constrained_states[0], new_state[0]))
    
    def test_get_constrained_actions(self):
        """Test getting constrained actions - should always return fixed shape tensor."""
        # Should return zero tensor with fixed shape by default
        constrained_actions = self.prev_chunk.get_constrained_actions()
        self.assertEqual(constrained_actions.shape, (self.action_num, self.action_dim))
        self.assertTrue(torch.all(constrained_actions == 0))
        
        # Test with pre-initialized actions
        test_actions = torch.randn(3, self.action_dim)
        prev_chunk_with_actions = PrevChunk(
            actions=test_actions,
            action_constrained_num=3,
            action_num=self.action_num,
            action_dim=self.action_dim,
            state_num=self.state_num,
            state_dim=self.state_dim
        )
        constrained_actions = prev_chunk_with_actions.get_constrained_actions()
        self.assertEqual(constrained_actions.shape, (self.action_num, self.action_dim))
        # Check that first 3 actions match
        self.assertTrue(torch.equal(constrained_actions[:3], test_actions))
    
    def test_get_action_prefix_weights(self):
        """Test generation of action prefix weights."""
        # Initially, no actions are constrained
        weights = self.prev_chunk.get_action_prefix_weights()
        self.assertEqual(weights.shape, (self.action_num,))
        self.assertTrue(torch.all(weights == 0))
        
        # Update action_constrained_num and test again
        self.prev_chunk.action_constrained_num = 5
        weights = self.prev_chunk.get_action_prefix_weights()
        expected = torch.zeros(self.action_num)
        expected[:5] = 1
        self.assertTrue(torch.equal(weights, expected))
    
    def test_get_state_prefix_weights(self):
        """Test generation of state prefix weights."""
        # Initially, no states are constrained
        weights = self.prev_chunk.get_state_prefix_weights()
        self.assertEqual(weights.shape, (self.state_num,))
        self.assertTrue(torch.all(weights == 0))
        
        # Update state_constrained_num and test again
        self.prev_chunk.state_constrained_num = 2
        weights = self.prev_chunk.get_state_prefix_weights()
        expected = torch.zeros(self.state_num)
        expected[:2] = 1
        self.assertTrue(torch.equal(weights, expected))
    
    def test_get_prefix_weights(self):
        """Test generation of combined prefix weights - should return (action_num + state_num,) tensor."""
        # Test with initial values (all zeros)
        weights = self.prev_chunk.get_prefix_weights()
        total_size = self.action_num + self.state_num
        self.assertEqual(weights.shape, (total_size,))
        self.assertTrue(torch.all(weights == 0))
        
        # Update constrained numbers and test
        self.prev_chunk.action_constrained_num = 3
        self.prev_chunk.state_constrained_num = 2
        weights = self.prev_chunk.get_prefix_weights()
        
        # Expected: first action_num elements have first action_constrained_num as 1
        #           last state_num elements have first state_constrained_num as 1
        expected = torch.zeros(total_size)
        # Action part: indices 0 to action_num-1, first action_constrained_num are 1
        expected[:self.prev_chunk.action_constrained_num] = 1
        # State part: indices action_num to action_num+state_num-1, first state_constrained_num are 1
        expected[self.action_num:self.action_num + self.prev_chunk.state_constrained_num] = 1
        
        self.assertTrue(torch.equal(weights, expected))
        
        # Additional verification with different values
        self.prev_chunk.state_constrained_num = 0
        self.prev_chunk.action_constrained_num = 0
        weights = self.prev_chunk.get_prefix_weights()
        self.assertTrue(torch.all(weights == 0))
        
        # Test with max constraints
        self.prev_chunk.state_constrained_num = self.state_num
        self.prev_chunk.action_constrained_num = self.action_num
        weights = self.prev_chunk.get_prefix_weights()
        expected_all_ones = torch.ones(total_size)
        self.assertTrue(torch.equal(weights, expected_all_ones))
    
    def test_get_constrain_mode(self):
        """Test getting the constrain mode."""
        self.assertEqual(self.prev_chunk.get_constrain_mode(), "Feedback")
        
        # Change constrain mode and test again
        self.prev_chunk.constrain_mode = "NoFeedback"
        self.assertEqual(self.prev_chunk.get_constrain_mode(), "NoFeedback")

    def test_single_thread_chunk_carry_over_lifecycle(self):
        """Validate the intended single-thread lifecycle across two chunks.

        Semantics:
        - next inference starts with previous action leftover as action prefix
        - state feedback is accumulated during execution of the current chunk
        - after the chunk is consumed by inference, state prefix is reset while
          action prefix can be replaced by the next leftover
        """
        leftover_actions = torch.randn(6, self.action_dim)
        prev_chunk = PrevChunk(
            actions=leftover_actions,
            action_constrained_num=6,
            action_num=self.action_num,
            action_dim=self.action_dim,
            state_num=self.state_num,
            state_dim=self.state_dim,
        )

        self.assertTrue(torch.equal(prev_chunk.actions[:6], leftover_actions))
        self.assertEqual(prev_chunk.action_constrained_num, 6)
        self.assertEqual(prev_chunk.state_constrained_num, 0)

        first_state = torch.randn(self.state_dim)
        second_state = torch.randn(self.state_dim)
        prev_chunk.append_new_state(first_state)
        prev_chunk.append_new_state(second_state)

        self.assertEqual(prev_chunk.state_constrained_num, 2)
        self.assertTrue(torch.equal(prev_chunk.states[0], first_state))
        self.assertTrue(torch.equal(prev_chunk.states[1], second_state))

        # Simulate "start next inference": keep action prefix but clear consumed state feedback.
        prev_chunk.states.zero_()
        prev_chunk.state_constrained_num = 0

        self.assertEqual(prev_chunk.action_constrained_num, 6)
        self.assertEqual(prev_chunk.state_constrained_num, 0)
        self.assertTrue(torch.equal(prev_chunk.actions[:6], leftover_actions))
        self.assertTrue(torch.all(prev_chunk.states == 0))

if __name__ == '__main__':
    unittest.main()
