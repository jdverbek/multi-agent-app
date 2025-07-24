import asyncio
from .main_controller import MainController
from .tasks import Task

async def main():
    controller = MainController()

    # Start the controller loop in the background
    controller_task = asyncio.create_task(controller.run())

    # Submit some example tasks
    await controller.submit_task(Task(type="analyseer code", content="print('hi')", role="CodeVerifier"))
    await controller.submit_task(Task(type="genereer script", content="data", role="Developer"))

    # Wait a short time for tasks to be processed
    await asyncio.sleep(0.5)

    # Since this is an example, we cancel the controller loop
    controller_task.cancel()
    try:
        await controller_task
    except asyncio.CancelledError:
        pass

if __name__ == "__main__":
    asyncio.run(main())
