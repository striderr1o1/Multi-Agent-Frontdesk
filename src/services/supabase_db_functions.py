from supabase import Client
from .supabase_client import get_supabase_client
# Database functions go here
async def get_namespacename_from_supabase(client: Client, user_id):
    response = await (client.table("pinecone_data_table")
                .select("namespace_name")
                .eq("business_id", user_id)
                .execute())
    # if response is None, raise error
    namespace_name = ""
    if response is not None and response.data:
        namespace_name = response.data[0]["namespace_name"]
    return namespace_name

#async def get_thread_id_from_supabase(client: Client, user_id):
#    response = await (client.table("links")
#                .select("thread_id")
#                .eq("business_id", user_id)
#                .execute()
#                )
#    thread_id = ""
#    if response is not None and response.data:
#        thread_id = response.data[0]["thread_id"]
#    return thread_id

async def get_published_status_from_supabase(client: Client, user_id):
    response = await (client.table("links")
                .select("published")
                .eq("business_id", user_id)
                .execute()
                )
    published = False
    if response is not None and response.data:
        published = response.data[0]["published"]
    return published

async def set_published_status_in_supabase(client: Client, user_id, published: bool):
    response = await (client.table("links")
                .update({"published": published})
                .eq("business_id", user_id)
                .execute()
                )
    if response is None or not response.data:
        raise ValueError(f"No links row found for user {user_id}")
    return response.data[0]["published"]

async def get_slots_from_supabase(client: Client, user_id):
    response = await (client.table("slots")
                .select("slotid, time_start, time_end, occupier_email, status")
                .eq("business_id", user_id)
                .execute()
                )
    slots = []
    if response is not None and response.data:
        slots = response.data
    return slots

async def get_url_from_supabase(client: Client, user_id):
    response = await (client.table("links")
                    .select("url")
                    .eq("business_id", user_id)
                    .execute()
                    )
    url= ""
    if response is not None and response.data:
        url= response.data[0]["url"]
    return url

async def get_publish_status_from_supabase(client: Client, user_id):
    response = await (client.table("links")
                .select("published")
                .eq("business_id", user_id)
                .execute()
                )
    status = False
    if response is not None and response.data:
        status = response.data[0]["published"]
    return status

async def confirm_booking_by_verification_id(client: Client, verification_id):
    response =await (client.rpc("confirm_verification", {"verf_id": verification_id})
                .execute())
    return response.data

async def insert_slot_into_supabase(client: Client, user_id, time_start, time_end):
    # the business is opening an empty slot, so it starts unoccupied and pending -
    # a customer booking through update_room_data is what fills occupier_email in
    response = await (client.table("slots")
                .insert({
                    "business_id": user_id,
                    "time_start": time_start,
                    "time_end": time_end,
                    "occupier_email": None,
                    "status": "pending",
                })
                .execute()
                )
    if response is None or not response.data:
        raise ValueError(f"Slot insert returned no row for user {user_id}")
    return response.data[0]

async def delete_slot_from_supabase(client: Client, user_id, slot_id):
    # business_id is matched as well as the primary key, so a slot belonging to
    # another business can't be deleted by guessing its slotid
    response = await (client.table("slots")
                .delete()
                .eq("slotid", slot_id)
                .eq("business_id", user_id)
                .execute()
                )
    # a delete that matched nothing still returns 200 with an empty list, so the
    # caller can't tell "deleted" from "not yours / not there" without this
    if response is None or not response.data:
        raise ValueError(f"No slot {slot_id} found for user {user_id}")
    return response.data[0]

async def get_pinecone_id_from_supabase(client: Client, user_id):
    # the ingestions row has to point at this business's pinecone_data_table row,
    # which is the same row get_namespacename_from_supabase reads the namespace from
    response = await (client.table("pinecone_data_table")
                .select("pc_id")
                .eq("business_id", user_id)
                .execute()
                )
    pc_id = ""
    if response is not None and response.data:
        pc_id = response.data[0]["pc_id"]
    return pc_id

async def insert_ingestion_into_supabase(client: Client, user_id, pc_id, source_name, record_ids):
    # business_id is written explicitly: the insert policy on public.ingestions
    # checks auth.uid() = business_id, so the column can't be left out
    response = await (client.table("ingestions")
                .insert({
                    "business_id": user_id,
                    "pc_id": pc_id,
                    "source_name": source_name,
                    "record_ids_json": record_ids,
                })
                .execute()
                )
    if response is None or not response.data:
        raise ValueError(f"Ingestion insert returned no row for user {user_id}")
    return response.data[0]

async def get_ingestions_from_supabase(client: Client, user_id):
    # record_ids_json is deliberately not selected - the dashboard only lists the
    # documents, and the vector id blob is large enough to be worth not shipping
    response = await (client.table("ingestions")
                .select("ing_id, source_name")
                .eq("business_id", user_id)
                .execute()
                )
    ingestions = []
    if response is not None and response.data:
        ingestions = response.data
    return ingestions

async def get_record_ids_from_supabase(client: Client, user_id, ingestion_id):
    # business_id is matched as well as the primary key, so another business's
    # vector ids can't be read (or later deleted) by guessing an ing_id
    response = await (client.table("ingestions")
                .select("record_ids_json")
                .eq("ing_id", ingestion_id)
                .eq("business_id", user_id)
                .execute()
                )
    record_ids = []
    if response is not None and response.data:
        stored = response.data[0]["record_ids_json"]
        # _record_ingestion writes {"vector_ids_list": [...]}, but a bare list is
        # accepted too so a row written before that wrapper still deletes cleanly
        if isinstance(stored, dict):
            record_ids = stored.get("vector_ids_list") or []
        elif isinstance(stored, list):
            record_ids = stored
    return record_ids

async def delete_ingestion_from_supabase(client: Client, user_id, ingestion_id):
    response = await (client.table("ingestions")
                .delete()
                .eq("ing_id", ingestion_id)
                .eq("business_id", user_id)
                .execute()
                )
    # same as delete_slot_from_supabase: a delete matching nothing is still a 200
    # with an empty list, so "deleted" and "not yours / not there" need separating
    if response is None or not response.data:
        raise ValueError(f"No ingestion {ingestion_id} found for user {user_id}")
    return response.data[0]

async def get_business_id_from_url_string(client: Client, url_string):
    #response = (client.table()) # get from auth table? match link id to business id?
    # maybe create a new policy, where (url_string = extracted_url_string)
    response = await (
            client.table("links")
            .select("business_id")
            .eq("url", url_string)
            .execute()
            )
    business_id = ""
    if response is not None and response.data:
        business_id = response.data[0]["business_id"]
    return business_id

async def get_customer_client_side_id(client: Client, user_id, customer_clientside_id):
    response = await (
            client.table("customers_data")
            .select("customer_client_side_id")
            .eq("customer_client_side_id", customer_clientside_id)
            .eq("business_id", user_id)
            .execute()
            )
    customer_cs_id = ""
    if response is not None and response.data:
        customer_cs_id = response.data[0]["customer_client_side_id"]
    return customer_cs_id


async def increment_customer_requests_in_db(client: Client, user_id, customer_client_side_id):
    # read-modify-write, not an atomic `total_requests = total_requests + 1`:
    # postgrest can't express that, so two concurrent queries from the same
    # customer can both read the same value and one increment is lost. An RPC
    # doing the update in SQL is the fix if that ever matters.
    # business_id is matched as well as the client side id, so one business
    # can't touch a counter belonging to another one's customer
    response = await (
            client.table("customers_data")
            .select("total_requests")
            .eq("customer_client_side_id", customer_client_side_id)
            .eq("business_id", user_id)
            .execute()
            )
    if response is None or not response.data:
        raise ValueError(f"No customers_data row for client side id {customer_client_side_id}")
    # the column is nullable, so a row written before the default was in place
    # (or with an explicit null) reads back as None rather than 0
    current = response.data[0]["total_requests"] or 0
    response = await (
            client.table("customers_data")
            .update({"total_requests": current + 1})
            .eq("customer_client_side_id", customer_client_side_id)
            .eq("business_id", user_id)
            .execute()
            )
    if response is None or not response.data:
        raise ValueError(f"total_requests not incremented for {customer_client_side_id}")
    return response.data[0]["total_requests"]


async def save_customer_client_side_id_in_db(client: Client, user_id, customer_client_side_id):
    response = await (
            client.table("customers_data")
            .insert({
                "customer_client_side_id": customer_client_side_id,
                "business_id": user_id
                }
                )
            .execute()
            )
    if response is None or not response.data:
        raise ValueError(f"client side id not saved")
    return response.data[0]

async def save_customer_chat(client: Client, user_id, customer_client_side_id, user_ai_chat):
    # read-modify-write, same as increment_customer_requests_in_db: postgrest can't
    # express `messages = messages || $1`, so the whole transcript is read back and
    # sent again on every turn. Two overlapping queries from the same customer can
    # both read the same list and one turn is lost. An RPC appending in SQL is the
    # fix if that ever matters.
    # business_id is matched as well as the client side id, so this doubles as the
    # ownership check - a client side id belonging to another business matches no
    # row and never gets written to
    response = await (
            client.table("customers_data")
            .select("messages")
            .eq("customer_client_side_id", customer_client_side_id)
            .eq("business_id", user_id)
            .execute()
            )
    if response is None or not response.data:
        raise ValueError(f"client id invalid")

    # the column is nullable, so a row that has never been written to (the one
    # save_customer_client_side_id_in_db inserts) reads back as None
    existing = response.data[0]["messages"] or []
    # a row written before this function appended rather than overwrote holds a
    # bare pair rather than a list of them, so it is carried over as the first entry
    if not isinstance(existing, list):
        existing = [existing]
    existing.append(user_ai_chat)

    response = await (
            client.table("customers_data")
            .update({"messages": existing})
            .eq("customer_client_side_id", customer_client_side_id)
            .eq("business_id", user_id)
            .execute()
            )
    if response is None or not response.data:
        raise ValueError(f"Could not update the customer chat")
    return response.data[0]["messages"]
