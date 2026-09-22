from langchain_core.tools import ToolException
from langchain.tools import tool
from langchain_core.runnables.config import RunnableConfig
from services.email_service import get_html_content, send_email

@tool
async def fetch_room_data(config: RunnableConfig):
    """Fetch all slots/rooms belonging to the current user's business."""
    try:
        user_id = config["configurable"]["user_id"]
        # per-request client, threaded via RunnableConfig; admits the authenticated role
        supabase = config["configurable"]["supabase_client"]
        response = await (supabase.table("slots")
        .select("slotid, time_start, time_end, occupier_email")
        .eq("business_id", user_id)
        .execute())
        return response
    except Exception as e:
        raise ToolException(f"Error in tool execution: {e}")

@tool
async def update_room_data(slot_id: str, occupier_email: str, config: RunnableConfig):
    """Assign an existing slot to an occupier, this tool sends them a confirmation link.

    The slot's times are set by the business and are not editable here - the only
    thing this changes is who the slot belongs to.

    Args:
        slot_id: slotid of the slot to assign, as returned by fetch_room_data
        occupier_email: email of the person taking the slot
    """
    try:
        user_id = config["configurable"]["user_id"]
        supabase = config["configurable"]["supabase_client"]
        response = await (supabase.table("slots")
                    .update({
                        "occupier_email": occupier_email,
                        "status": "pending"
                    })
                    .eq("slotid", slot_id)
                    .eq("business_id", user_id)
                    .execute())
        # an update matching nothing still comes back 200 with an empty list, so
        # without this a bad slotid would silently "succeed" and send no email
        if not response.data:
            raise ValueError(f"No slot {slot_id} belonging to this business") # times come from the row rather than the agent, so the email can only
        # describe the slot that was actually assigned
        slot = response.data[0]
        verf_id = slot["verification_id"]
        html_content = get_html_content(occupier_email, slot["time_start"], slot["time_end"], verf_id)
        await send_email(occupier_email, "Booking Verification", html_content)
        return response
    except Exception as e:
        raise ToolException(f"Error in tool execution: {e}")

@tool
def delete_room_data(slot_id: int, config: RunnableConfig):
    """Delete a slot/room from the database by its id.

    Args:
        slot_id: the id of the slot to delete
    """
    #try:
    #    user_id = config["configurable"]["user_id"]
    #    response = (supabase.table("slots")
    #        .delete()
    #        .eq("id", slot_id)
    #        .eq("business_id", user_id)
    #        .execute())
    #    return response
    #except Exception as e:
    #    raise ToolException(f"Error in tool execution: {e}")
    return

@tool
async def insert_room_data(time_start: str, time_end: str, occupier_email: str, config: RunnableConfig):
    """ inserts data into the database with a pending, and sends an email to the provided email for booking confirmation.

    Args:
        time_start: reservation start datetime (ISO 8601 string, e.g. '2026-07-31T09:00:00+00:00')
        time_end: reservation end datetime (ISO 8601 string)
        occupier_email: email of the person occupying the slot
    """
    try:
        user_id = config["configurable"]["user_id"]
        supabase = config["configurable"]["supabase_client"]
        response = await (supabase.table("slots")
            .insert({
                "business_id": user_id,
                "time_start": time_start,
                "time_end": time_end,
                "occupier_email": occupier_email,
                "status": "pending"
            })
            .execute())
        verf_id = response.data[0]["verification_id"]
        html_content = get_html_content(occupier_email, time_start, time_end, verf_id)
        await send_email(occupier_email, "Booking Verification", html_content) #embed a webhook link too in the email
        return response
    except Exception as e:
        raise ToolException(f"Error in tool execution: {e}")
