from api.agents.rag.rag import (
    process_context,
    retrieve_data,
    process_reviews_context,
    retrieve_reviews_data,
)
from qdrant_client.models import (
    Prefetch,
    FusionQuery,
    Filter,
    FieldCondition,
    MatchValue,
)
from api.core.config import DEFAULT_TOP_K
from api.server.models import RAGRequestExtraOptions
from api.utils.tracing import hide_sensitive_inputs
from langsmith import traceable
from numpy import zeros
from api.core.qdrant import qdrant_client
from api.core.config import RAG_COLLECTIONS, RAG_EMBEDDING_MODEL
from api.core.pg import postgres_client
from api.utils.logs import logger


@traceable(
    process_inputs=hide_sensitive_inputs,
    name="get_formatted_item_context",
    run_type="tool",
)
def get_formatted_item_context(
    query: str, top_k: int = DEFAULT_TOP_K, enable_reranking: bool = False
) -> str:
    """Get the top k context, each representing an inventory item for a given query.

    Args:
        query: The query to get the top k context for
        top_k: The number of context chunks to retrieve, works best with 5 or more
        enable_reranking: Whether to enable reranking of retrieved results based on relevance to the query. Default is false.

    Returns:
        A string of the top k context chunks with IDs and average ratings prepending each chunk, each representing an inventory item for a given query.
    """
    extra_options = RAGRequestExtraOptions(
        top_k=top_k,
        enable_reranking=enable_reranking,
    )

    context = retrieve_data(query, extra_options)
    formatted_context = process_context(context)

    return formatted_context


@traceable(
    process_inputs=hide_sensitive_inputs,
    name="get_formatted_reviews_context",
    run_type="tool",
)
def get_formatted_reviews_context(query: str, item_list: list, top_k: int = 15) -> str:
    """Get the top k reviews matching a query for a list of prefiltered items.

    Args:
        query: The query to get the top k reviews for
        item_list: The list of item IDs to prefilter for before running the query
        top_k: The number of reviews to retrieve, this should be at least 20 if multiple items are prefiltered

    Returns:
        A string of the top k context chunks with IDs prepending each chunk, each representing a review for a given inventory item for a given query.
    """

    context = retrieve_reviews_data(query, item_list, top_k)
    formatted_context = process_reviews_context(context)

    return formatted_context


### Shopping Cart Tools

tools_database_name = "tools_database"
"""Name of the Postgres database to use for the shopping cart tools."""


@traceable(name="add_to_shopping_cart", run_type="tool")
def add_to_shopping_cart(items: list[dict], user_id: str, cart_id: str) -> str:
    """Add a list of provided items to the shopping cart.

    Args:
        items: A list of items to add to the shopping cart. Each item is a dictionary with the following keys: product_id, quantity.
        user_id: The id of the user to add the items to the shopping cart.
        cart_id: The id of the shopping cart to add the items to.

    Returns:
        A list of the items added to the shopping cart.
    """

    try:
        with postgres_client.get(db=tools_database_name) as cursor:
            for item in items:
                product_id = item["product_id"]
                quantity = item["quantity"]

                prefetch: Prefetch = Prefetch(
                    query=zeros(1536).tolist(),
                    filter=Filter(
                        must=[
                            FieldCondition(
                                key="parent_asin",
                                match=MatchValue(value=product_id),
                            )
                        ]
                    ),
                    using=RAG_EMBEDDING_MODEL,
                    limit=20,
                )

                payload = (
                    qdrant_client.get()
                    .query_points(
                        collection_name=RAG_COLLECTIONS["items"],
                        prefetch=[prefetch],
                        query=FusionQuery(fusion="rrf"),
                        limit=1,
                    )
                    .points[0]
                    .payload
                )

                product_image_url = payload.get("image")
                price = payload.get("price")
                currency = "USD"

                # Check if item already exists
                check_query = """
                    SELECT id, quantity, price 
                    FROM shopping_cart_items 
                    WHERE user_id = %s AND shopping_cart_id = %s AND product_id = %s
                """
                cursor.execute(check_query, (user_id, cart_id, product_id))
                existing_item = cursor.fetchone()

                if existing_item:
                    # Update existing item
                    new_quantity = existing_item["quantity"] + quantity

                    update_query = """
                        UPDATE shopping_cart_items 
                        SET 
                            quantity = %s,
                            price = %s,
                            currency = %s,
                            product_image_url = COALESCE(%s, product_image_url)
                        WHERE user_id = %s AND shopping_cart_id = %s AND product_id = %s
                        RETURNING id, quantity, price
                    """

                    cursor.execute(
                        update_query,
                        (
                            new_quantity,
                            price,
                            currency,
                            product_image_url,
                            user_id,
                            cart_id,
                            product_id,
                        ),
                    )

                else:
                    # Insert new item
                    insert_query = """
                        INSERT INTO shopping_cart_items (
                            user_id, shopping_cart_id, product_id,
                            price, quantity, currency, product_image_url
                        ) VALUES (%s, %s, %s, %s, %s, %s, %s)
                        RETURNING id, quantity, price
                    """

                    cursor.execute(
                        insert_query,
                        (
                            user_id,
                            cart_id,
                            product_id,
                            price,
                            quantity,
                            currency,
                            product_image_url,
                        ),
                    )

        return f"Added {items} to the shopping cart."
    except Exception as e:
        postgres_client.close()  # Close the connection on error to reset the state
        logger.error(
            "Error occurred while adding items to the shopping cart", exc_info=True
        )
        raise e


@traceable(name="get_shopping_cart", run_type="tool")
def get_shopping_cart(user_id: str, cart_id: str) -> list[dict]:
    """
    Retrieve all items in a user's shopping cart.

    Args:
        user_id: User ID
        cart_id: Cart identifier

    Returns:
        List of dictionaries containing cart items
    """
    try:
        with postgres_client.get(db=tools_database_name) as cursor:

            query = """
                    SELECT 
                        product_id, price, quantity,
                        currency, product_image_url,
                        (price * quantity) as total_price
                    FROM shopping_cart_items 
                    WHERE user_id = %s AND shopping_cart_id = %s
                    ORDER BY added_at DESC
                """
            cursor.execute(query, (user_id, cart_id))

            return [dict(row) for row in cursor.fetchall()]
    except Exception as e:
        postgres_client.close()  # Close the connection on error to reset the state
        logger.error("Error occurred while retrieving the shopping cart", exc_info=True)
        raise e


@traceable(name="remove_from_cart", run_type="tool")
def remove_from_cart(product_id: str, user_id: str, cart_id: str) -> str:
    """
    Remove an item completely from the shopping cart.

    Args:
        user_id: User ID
        product_id: Product ID to remove
        cart_id: Cart identifier

    Returns:
        True if item was removed, False if item wasn't found
    """
    try:
        with postgres_client.get(db=tools_database_name) as cursor:

            query = """
                    DELETE FROM shopping_cart_items
                    WHERE user_id = %s AND shopping_cart_id = %s AND product_id = %s
                """
            cursor.execute(query, (user_id, cart_id, product_id))

            return cursor.rowcount > 0
    except Exception as e:
        postgres_client.close()  # Close the connection on error to reset the state
        logger.error(
            "Error occurred while removing item from the shopping cart", exc_info=True
        )
        raise e


### Warehouse Manager Agent Tools


@traceable(name="check_warehouse_availability", run_type="tool")
def check_warehouse_availability(items: list[dict]) -> dict:
    """Check availability of items across warehouses, including partial fulfillment options.

    Args:
        items: A list of items to check. Each item is a dictionary with keys: product_id, quantity.

    Returns:
        A dictionary containing:
        - can_fulfill_completely: bool indicating if all items can be fulfilled from at least one warehouse
        - warehouses_full_fulfillment: list of warehouses that can fulfill the entire order
        - warehouses_partial_fulfillment: list of warehouses with partial availability
        - unavailable_items: list of items that cannot be fulfilled from any warehouse
        - details: detailed breakdown per warehouse with availability for each item
    """

    try:
        with postgres_client.get(db=tools_database_name) as cursor:
            result = {
                "can_fulfill_completely": False,
                "warehouses_full_fulfillment": [],
                "warehouses_partial_fulfillment": [],
                "unavailable_items": [],
                "details": [],
            }

            # Check each warehouse for availability
            warehouse_query = """
                SELECT DISTINCT warehouse_id, warehouse_name, warehouse_location
                FROM inventory
            """
            cursor.execute(warehouse_query)
            warehouses = cursor.fetchall()

            for warehouse in warehouses:
                warehouse_can_fulfill_all = True
                has_any_availability = False
                warehouse_details = {
                    "warehouse_id": warehouse["warehouse_id"],
                    "warehouse_name": warehouse["warehouse_name"],
                    "warehouse_location": warehouse["warehouse_location"],
                    "items": [],
                    "can_fulfill_all": False,
                    "has_partial": False,
                }

                for item in items:
                    product_id = item["product_id"]
                    requested_quantity = item["quantity"]

                    # Check availability in this warehouse
                    availability_query = """
                        SELECT product_id, total_quantity, reserved_quantity, available_quantity
                        FROM inventory
                        WHERE warehouse_id = %s AND product_id = %s
                    """
                    cursor.execute(
                        availability_query, (warehouse["warehouse_id"], product_id)
                    )
                    inventory = cursor.fetchone()

                    available_qty = inventory["available_quantity"] if inventory else 0

                    item_detail = {
                        "product_id": product_id,
                        "requested": requested_quantity,
                        "available": available_qty,
                        "can_fulfill_completely": available_qty >= requested_quantity,
                        "can_fulfill_partially": available_qty > 0
                        and available_qty < requested_quantity,
                    }

                    warehouse_details["items"].append(item_detail)

                    # Track if warehouse can fulfill this item completely
                    if available_qty < requested_quantity:
                        warehouse_can_fulfill_all = False

                    # Track if warehouse has any availability for any item
                    if available_qty > 0:
                        has_any_availability = True

                # Categorize warehouse
                if warehouse_can_fulfill_all:
                    warehouse_details["can_fulfill_all"] = True
                    result["warehouses_full_fulfillment"].append(
                        {
                            "warehouse_id": warehouse["warehouse_id"],
                            "warehouse_name": warehouse["warehouse_name"],
                            "warehouse_location": warehouse["warehouse_location"],
                        }
                    )
                elif has_any_availability:
                    warehouse_details["has_partial"] = True
                    result["warehouses_partial_fulfillment"].append(
                        {
                            "warehouse_id": warehouse["warehouse_id"],
                            "warehouse_name": warehouse["warehouse_name"],
                            "warehouse_location": warehouse["warehouse_location"],
                        }
                    )

                result["details"].append(warehouse_details)

            # Check if any items cannot be fulfilled from any warehouse
            for item in items:
                product_id = item["product_id"]
                requested_quantity = item["quantity"]

                # Get total available quantity across all warehouses
                total_available_query = """
                    SELECT product_id, SUM(available_quantity) as total_available
                    FROM inventory
                    WHERE product_id = %s
                    GROUP BY product_id
                """
                cursor.execute(total_available_query, (product_id,))
                total_available = cursor.fetchone()

                total_available_qty = (
                    total_available["total_available"] if total_available else 0
                )

                if total_available_qty < requested_quantity:
                    result["unavailable_items"].append(
                        {
                            "product_id": product_id,
                            "requested": requested_quantity,
                            "total_available_across_warehouses": total_available_qty,
                            "shortage": requested_quantity - total_available_qty,
                        }
                    )

            result["can_fulfill_completely"] = (
                len(result["warehouses_full_fulfillment"]) > 0
                and len(result["unavailable_items"]) == 0
            )

            return result

    finally:
        postgres_client.close()


@traceable(name="reserve_warehouse_items", run_type="tool")
def reserve_warehouse_items(reservations: list[dict]) -> dict:
    """Reserve items from multiple warehouses in a single transaction.

    Args:
        reservations: A list of reservations. Each reservation is a dictionary with keys:
                     - warehouse_id: The warehouse to reserve from
                     - product_id: The product to reserve
                     - quantity: The quantity to reserve

    Returns:
        A dictionary containing:
        - success: bool indicating if all reservations were successful
        - reserved_items: list of successfully reserved items
        - failed_items: list of items that could not be reserved
    """

    try:
        with postgres_client.get(db=tools_database_name, autocommit=False) as cursor:
            result = {"success": False, "reserved_items": [], "failed_items": []}

            for reservation in reservations:
                warehouse_id = reservation["warehouse_id"]
                product_id = reservation["product_id"]
                quantity = reservation["quantity"]

                # Check and lock the inventory row
                check_query = """
                    SELECT warehouse_id, product_id, warehouse_name, warehouse_location, 
                           total_quantity, reserved_quantity, available_quantity
                    FROM inventory
                    WHERE warehouse_id = %s AND product_id = %s
                    FOR UPDATE
                """
                cursor.execute(check_query, (warehouse_id, product_id))
                inventory = cursor.fetchone()

                if inventory and inventory["available_quantity"] >= quantity:
                    # Update inventory to reserve the items
                    update_query = """
                        UPDATE inventory
                        SET reserved_quantity = reserved_quantity + %s
                        WHERE warehouse_id = %s AND product_id = %s
                    """
                    cursor.execute(update_query, (quantity, warehouse_id, product_id))

                    result["reserved_items"].append(
                        {
                            "product_id": product_id,
                            "quantity": quantity,
                            "warehouse_id": warehouse_id,
                            "warehouse_name": inventory["warehouse_name"],
                            "warehouse_location": inventory["warehouse_location"],
                        }
                    )
                else:
                    result["failed_items"].append(
                        {
                            "product_id": product_id,
                            "warehouse_id": warehouse_id,
                            "requested": quantity,
                            "available": (
                                inventory["available_quantity"] if inventory else 0
                            ),
                            "reason": (
                                "insufficient_stock"
                                if inventory
                                else "not_in_warehouse"
                            ),
                        }
                    )

            # Only commit if all items were successfully reserved
            if len(result["failed_items"]) == 0:
                postgres_client.commit()
                result["success"] = True
            else:
                postgres_client.rollback()
                result["success"] = False

            return result

    except Exception as e:
        postgres_client.rollback()
        raise e
    finally:
        postgres_client.close()
