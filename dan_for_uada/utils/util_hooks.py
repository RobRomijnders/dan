from tensorflow.python.training.basic_session_run_hooks import SecondOrStepTimer
from tensorflow.python.training.basic_session_run_hooks import CheckpointSaverHook


class SecondOrStepTimerCustom(SecondOrStepTimer):
    def __init__(self, start_step=10, *args, **kwargs):
        super(SecondOrStepTimerCustom, self).__init__(*args, **kwargs)
        self.start_step = start_step

    def should_trigger_for_step(self, step):
        """Return true if the timer should trigger for the specified step.

        Args:
          step: Training step to trigger on.

        Returns:
          True if the difference between the current time and the time of the last
          trigger exceeds `every_secs`, or if the difference between the current
          step and the last triggered step exceeds `every_steps`. False otherwise.
        """
        if self._last_triggered_step is None:
            return True

        if self._last_triggered_step == step:
            return False

        if self._every_secs is not None:
            assert False, 'Only implemented for triggers based on steps'

        if self._every_steps is not None:
            if step > self.start_step:
                if step >= self._last_triggered_step + self._every_steps:
                    return True

        return False


class CheckpointSaverHookCustom(CheckpointSaverHook):
    """Saves checkpoints every N steps or seconds."""

    def __init__(self, start_step=10, *args, **kwargs):
        super(CheckpointSaverHookCustom, self).__init__(*args, **kwargs)

        self._timer = SecondOrStepTimerCustom(start_step=start_step, every_steps=kwargs.get('save_steps'))
