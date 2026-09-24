from datetime import datetime
from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from dependencies import run_inference_with_stream
from services.auth_logic import check_session_exists
from utils.exceptions import BadRequestError
from services.supabase_db_functions import (
    get_url_from_supabase,
    get_published_status_from_supabase,
    set_published_status_in_supabase,
    get_slots_from_supabase,
    get_business_id_from_url_string,
    confirm_booking_by_verification_id,
    insert_slot_into_supabase,
    delete_slot_from_supabase,
    get_ingestions_from_supabase,
    get_namespacename_from_supabase,
    get_record_ids_from_supabase,
    delete_ingestion_from_supabase,
    get_customer_client_side_id,
    save_customer_client_side_id_in_db,
    increment_customer_requests_in_db
)
from KnowledgeBaseTool.ingestion import Ingestion

from services.supabase_client import get_supabase_client_with_token, get_supabase_anon_client
router = APIRouter()


@router.get("/get-url")
async def get_url(user: dict = Depends(check_session_exists)):
    try:
        user_id = user["id"]
        access_token = user["access_token"]
        supabase_client = await get_supabase_client_with_token(access_token)
        # might need to go and change database functions to suppport async I/O
        url = await get_url_from_supabase(supabase_client, user_id)
        published = await get_published_status_from_supabase(supabase_client, user_id)
        return {
                "url": url,
                "published": published
                }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal Error: {e}")


class PublishStatus(BaseModel):
    published: bool


class QueryRequest(BaseModel):
    query: str
    unique_id: str


class SlotCreation(BaseModel):
    # parsed as datetimes so a malformed value is a 422 here rather than a
    # Postgres error later; handed to the db function as ISO 8601 strings,
    # which is what the timestamptz columns take
    time_start: datetime
    time_end: datetime


class SlotDeletion(BaseModel):
    # `slotid` is the uuid primary key on the slots table, as used by update_room_data
    slot_id: str


class IngestionDeletion(BaseModel):
    # `ing_id` is the uuid primary key on the ingestions table; source_name comes
    # along for the response and the log, the row is matched on the id
    ingestion_id: str
    source_name: str | None = None


@router.post("/set-publish")
async def set_publish(status: PublishStatus, user: dict = Depends(check_session_exists)):
    try:
        user_id = user["id"]
        access_token = user["access_token"]
        supabase_client = await get_supabase_client_with_token(access_token)
        published = await set_published_status_in_supabase(supabase_client, user_id, status.published)
        return {"published": published}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal Error: {e}")


@router.get("/get-slots-data")
async def get_slots_data(user: dict = Depends(check_session_exists)):
    try:
        user_id = user["id"]
        access_token = user["access_token"]
        supabase_client = await get_supabase_client_with_token(access_token)
        slots = await get_slots_from_supabase(supabase_client, user_id)
        return {"slots": slots}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal Error: {e}")

@router.post("/add-slot")
async def create_slot(creation: SlotCreation, user: dict = Depends(check_session_exists)):
    try:
        user_id = user["id"]
        access_token = user["access_token"]
        supabase_client = await get_supabase_client_with_token(access_token)
        slot = await insert_slot_into_supabase(
            supabase_client,
            user_id,
            creation.time_start.isoformat(),
            creation.time_end.isoformat(),
        )
        return {"slot": slot}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal Error: {e}")


@router.post("/delete-slot")
async def delete_slot(deletion: SlotDeletion, user: dict = Depends(check_session_exists)):
    try:
        user_id = user["id"]
        access_token = user["access_token"]
        supabase_client = await get_supabase_client_with_token(access_token)
        deleted = await delete_slot_from_supabase(supabase_client, user_id, deletion.slot_id)
        return {"deleted": deleted}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal Error: {e}")


@router.get("/get-record-count")
async def get_record_count(user: dict = Depends(check_session_exists)):
    # one entry per ingested document now, from the ingestions table - the old
    # pinecone-backed count returned one entry per chunk, which the dashboard
    # can't show as a document list
    try:
        user_id = user["id"]
        access_token = user["access_token"]
        supabase_client = await get_supabase_client_with_token(access_token)
        ingestions = await get_ingestions_from_supabase(supabase_client, user_id)
        return [
            {"ingestion_id": row["ing_id"], "source_name": row["source_name"]}
            for row in ingestions
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal Error: {e}")


@router.post("/delete-ingested-source")
async def delete_ingested_source(deletion: IngestionDeletion, user: dict = Depends(check_session_exists)):
    # the vectors go first: if pinecone fails the row survives, so the ids are
    # still on record and the delete can be retried. Dropping the row first would
    # strand the vectors in the namespace with nothing left pointing at them.
    try:
        user_id = user["id"]
        access_token = user["access_token"]
        supabase_client = await get_supabase_client_with_token(access_token)
        record_ids = await get_record_ids_from_supabase(supabase_client, user_id, deletion.ingestion_id)
        namespace_name = await get_namespacename_from_supabase(supabase_client, user_id)
        ingestion_obj = Ingestion(supabase_client, user_id)
        vectors_deleted = await ingestion_obj.delete_ingestion_source(record_ids, namespace_name)
        await delete_ingestion_from_supabase(supabase_client, user_id, deletion.ingestion_id)
        return {
            "deleted": {
                "ingestion_id": deletion.ingestion_id,
                "source_name": deletion.source_name,
                "vectors_deleted": vectors_deleted,
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal Error: {e}")


@router.post("/c/query-agent/{url_string}")
async def customer_query(url_string: str, inf: QueryRequest):
    try:
        client = await get_supabase_anon_client()
        business_id = await get_business_id_from_url_string(client, url_string)
        publish_status = await get_published_status_from_supabase(client, business_id)
        if publish_status is not True:
            raise BadRequestError("URL not published")
        user = {"id": business_id}
        customer_cs_id = await get_customer_client_side_id(client, business_id, inf.unique_id)
        # saves the client side id, but doesnt save messages
        if len(customer_cs_id) == 0:
            response = await save_customer_client_side_id_in_db(client, business_id, inf.unique_id)
        # counted before the stream starts: StreamingResponse returns immediately
        # and the generator runs after this handler is gone, so an increment
        # placed after it would never be reached
        await increment_customer_requests_in_db(client, business_id, inf.unique_id)
        return StreamingResponse(
            run_inference_with_stream(inf.query, user, client, inf.unique_id, admin=False),
            media_type="text/event-stream",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal Error: {e}")


@router.get("/booking-confirmation/{verification_id}")
async def booking_confirmation(verification_id: str):
    try:
        client = await get_supabase_anon_client()
        await confirm_booking_by_verification_id(client, verification_id)
        return {"message": "Booking confirmed"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal Error: {e}")

