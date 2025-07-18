# interactive_session.py
from core.agent_loop import ask

print(
    ask("For which L does the pendulum's period stop increasing? "
        "If data are insufficient, tell me NEXT_EXPERIMENT points.")
)

print(
    ask("Does the relation period vs L look linear anywhere? Show the plot.")
)
