import React from 'react';

// Types for MCP Tool Results (shared with your server code)
interface ToolResult {
  content: any[];
  structured_content: any;
}

// Multi Dish Lookup Types
interface DishLookupData {
  dish_id: string;
  name_of_food: string;
  type_of_food: string;
  current_price: number;
  main_ingredients: string;
  how_to_prepare: string;
  taste: string;
  outstanding_fragrance: string;
  number_of_people_eating: number;
  combine_info: string;
}

interface MultiDishLookupProps {
  found_dishes: DishLookupData[];
  missing_ids: string[];
  errors: { dish_id: string; error: string }[];
  total_requested: number;
  total_found: number;
}

// Take Order Types
interface OrderedDish {
  dish_id: string;
  name: string;
  quantity: number;
  price: number;
  subtotal: number;
}

interface OrderData {
  order_id: string;
  customer_id: string;
  customer_name: string;
  total_bill: number;
  is_takeaway: boolean;
  table_id?: string;
  notes?: string;
  created_at?: string;
  ordered_dishes: OrderedDish[];
}

interface CustomerData {
  customer_id: string;
  full_name: string;
  phone?: string;
  email?: string;
}

interface TakeOrderProps {
  order: OrderData;
  customer: CustomerData;
  status: "success" | "failed";
  timestamp: string;
}

// Menu Value Count and Price Types
interface DishMenuItem {
  dish_id: string;
  name_of_food: string;
  current_price: number;
  type_of_food: string;
}

interface MenuGroup {
  type_name: string;
  count: number;
  dishes: DishMenuItem[];
}

interface MenuValueCountAndPriceProps {
  menu_groups: Record<string, MenuGroup>;
  total_dishes: number;
  total_groups: number;
  timestamp: string;
}

// LangGraph Generative UI Components for MCP Tools

const MultiDishLookupResult: React.FC<MultiDishLookupProps> = ({
  found_dishes,
  missing_ids,
  errors,
  total_requested,
  total_found
}) => {
  return (
    <div className="space-y-6 p-4 bg-white rounded-lg shadow-sm">
      <div className="flex items-center gap-4 flex-wrap">
        <div className="text-center p-3 bg-blue-50 rounded-lg min-w-[120px]">
          <div className="text-2xl font-bold text-blue-600">{total_requested}</div>
          <div className="text-sm text-blue-700">Requested</div>
        </div>
        <div className="text-center p-3 bg-green-50 rounded-lg min-w-[120px]">
          <div className="text-2xl font-bold text-green-600">{total_found}</div>
          <div className="text-sm text-green-700">Found</div>
        </div>
        <div className="text-center p-3 bg-yellow-50 rounded-lg min-w-[120px]">
          <div className="text-2xl font-bold text-yellow-600">{missing_ids?.length || 0}</div>
          <div className="text-sm text-yellow-700">Missing</div>
        </div>
      </div>

      {found_dishes && found_dishes.length > 0 && (
        <div>
          <h3 className="text-lg font-semibold text-gray-900 mb-3">Found Dishes</h3>
          <div className="grid gap-4">
            {found_dishes.map((dish) => (
              <div key={dish.dish_id} className="p-4 border rounded-lg bg-gray-50">
                <div className="flex justify-between items-start mb-2">
                  <h4 className="font-semibold text-gray-900">{dish.name_of_food}</h4>
                  <span className="font-bold text-green-600">
                    {dish.current_price?.toLocaleString('vi-VN')} vnđ
                  </span>
                </div>
                <div className="text-sm text-gray-600 space-y-1">
                  <div><strong>ID:</strong> {dish.dish_id}</div>
                  {dish.type_of_food && <div><strong>Type:</strong> {dish.type_of_food}</div>}
                  {dish.main_ingredients && (
                    <div><strong>Ingredients:</strong> {dish.main_ingredients}</div>
                  )}
                  {dish.taste && <div><strong>Taste:</strong> {dish.taste}</div>}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {missing_ids && missing_ids.length > 0 && (
        <div>
          <h3 className="text-lg font-semibold text-gray-900 mb-3">Not Found</h3>
          <div className="flex flex-wrap gap-2">
            {missing_ids.map((id) => (
              <span key={id} className="px-3 py-1 bg-yellow-100 text-yellow-800 rounded-full text-sm">
                {id}
              </span>
            ))}
          </div>
        </div>
      )}

      {errors && errors.length > 0 && (
        <div>
          <h3 className="text-lg font-semibold text-red-600 mb-3">Errors</h3>
          <div className="space-y-2">
            {errors.map((error, index) => (
              <div key={index} className="p-3 bg-red-50 border border-red-200 rounded-lg">
                <div className="text-red-700 text-sm">
                  <strong>{error.dish_id}:</strong> {error.error}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};

const MenuValueCountAndPriceResult: React.FC<MenuValueCountAndPriceProps> = ({
  menu_groups,
  total_dishes,
  total_groups,
  timestamp
}) => {
  const menuGroups = Object.values(menu_groups || {}) as MenuGroup[];
  const sortedGroups = menuGroups.sort((a: MenuGroup, b: MenuGroup) => a.type_name.localeCompare(b.type_name));

  // Calculate price statistics
  const allDishes = menuGroups.flatMap((g: MenuGroup) => g.dishes);
  const prices = allDishes.map(d => d.current_price);
  const minPrice = Math.min(...prices);
  const maxPrice = Math.max(...prices);
  const avgPrice = prices.reduce((a, b) => a + b, 0) / prices.length;

  return (
    <div className="space-y-6 p-4 bg-white rounded-lg shadow-sm">
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <div className="text-center p-3 bg-blue-50 rounded-lg">
          <div className="text-xl font-bold text-blue-600">{total_dishes}</div>
          <div className="text-sm text-blue-700">Total Dishes</div>
        </div>
        <div className="text-center p-3 bg-purple-50 rounded-lg">
          <div className="text-xl font-bold text-purple-600">{total_groups}</div>
          <div className="text-sm text-purple-700">Categories</div>
        </div>
        <div className="text-center p-3 bg-green-50 rounded-lg">
          <div className="text-xl font-bold text-green-600">{minPrice?.toLocaleString('vi-VN')}k</div>
          <div className="text-sm text-green-700">Lowest Price</div>
        </div>
        <div className="text-center p-3 bg-orange-50 rounded-lg">
          <div className="text-xl font-bold text-orange-600">{maxPrice?.toLocaleString('vi-VN')}k</div>
          <div className="text-sm text-orange-700">Highest Price</div>
        </div>
      </div>

      {sortedGroups.map((group) => (
        <div key={group.type_name}>
          <div className="flex items-center justify-between mb-3">
            <h3 className="text-lg font-semibold text-gray-900">{group.type_name}</h3>
            <span className="text-sm text-gray-600 bg-gray-100 px-2 py-1 rounded">
              {group.count} items
            </span>
          </div>
          <div className="grid gap-2">
            {group.dishes.map((dish) => (
              <div key={dish.dish_id} className="flex justify-between items-center p-3 bg-gray-50 rounded border">
                <div className="flex items-center gap-3">
                  <code className="text-sm text-gray-600 font-mono">{dish.dish_id}</code>
                  <span className="font-medium text-gray-900">{dish.name_of_food}</span>
                </div>
                <span className="font-bold text-green-600">
                  {dish.current_price?.toLocaleString('vi-VN')}k
                </span>
              </div>
            ))}
          </div>
        </div>
      ))}

      {timestamp && (
        <div className="text-center text-sm text-gray-500 mt-4 pt-4 border-t">
          Updated: {new Date(timestamp).toLocaleString('vi-VN')}
        </div>
      )}
    </div>
  );
};

const TakeOrderResult: React.FC<TakeOrderProps> = ({
  order,
  customer,
  status,
  timestamp
}) => {
  const isError = status === 'failed';

  if (isError) {
    return (
      <div className="p-4 bg-red-50 border border-red-200 rounded-lg">
        <h3 className="text-lg font-semibold text-red-800 mb-2">Order Failed</h3>
        <p className="text-red-700">Unable to process the order. Please try again.</p>
        <div className="text-sm text-red-600 mt-2">
          Timestamp: {new Date(timestamp).toLocaleString('vi-VN')}
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-6 p-4 bg-white rounded-lg shadow-lg">
      {/* Status Header */}
      <div className="text-center py-4 bg-green-50 rounded-lg border border-green-200">
        <div className="text-green-800 font-medium text-lg">✓ Order Placed Successfully</div>
        <div className="text-green-700 text-sm mt-1">
          Order ID: <code className="font-mono">{order.order_id}</code>
        </div>
        {order.created_at && (
          <div className="text-green-600 text-xs mt-1">
            {new Date(order.created_at).toLocaleString('vi-VN')}
          </div>
        )}
      </div>

      {/* Order Summary */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <div className="text-center p-3 bg-blue-50 rounded-lg">
          <div className="text-xl font-bold text-blue-600">
            {order.total_bill?.toLocaleString('vi-VN')}k
          </div>
          <div className="text-sm text-blue-700">Total Amount</div>
        </div>
        <div className="text-center p-3 bg-purple-50 rounded-lg">
          <div className="text-xl font-bold text-purple-600">
            {order.is_takeaway ? 'Takeaway' : 'Dine-in'}
          </div>
          <div className="text-sm text-purple-700">
            {order.table_id ? `Table ${order.table_id}` : 'Service Type'}
          </div>
        </div>
        <div className="text-center p-3 bg-orange-50 rounded-lg">
          <div className="text-xl font-bold text-orange-600">
            {order.ordered_dishes?.length || 0}
          </div>
          <div className="text-sm text-orange-700">Items Ordered</div>
        </div>
      </div>

      {/* Customer Details */}
      <div>
        <h3 className="text-lg font-semibold text-gray-900 mb-3">Customer Information</h3>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div className="space-y-2">
            <div><strong>ID:</strong> <code className="text-sm">{customer.customer_id}</code></div>
            <div><strong>Name:</strong> {customer.full_name}</div>
          </div>
          <div className="space-y-2">
            {customer.phone && <div><strong>Phone:</strong> {customer.phone}</div>}
            {customer.email && <div><strong>Email:</strong> {customer.email}</div>}
          </div>
        </div>
      </div>

      {/* Order Items */}
      <div>
        <h3 className="text-lg font-semibold text-gray-900 mb-3">Order Details</h3>
        <div className="bg-gray-50 rounded-lg overflow-hidden">
          <div className="px-4 py-3 bg-gray-100 border-b">
            <div className="font-medium text-gray-900">Items Ordered</div>
          </div>
          <div className="divide-y">
            {order.ordered_dishes?.map((dish, index) => (
              <div key={dish.dish_id} className="px-4 py-3">
                <div className="flex justify-between items-center">
                  <div className="flex-1">
                    <div className="flex items-center gap-3">
                      <code className="text-sm text-gray-600 font-mono">{dish.dish_id}</code>
                      <span className="font-medium text-gray-900">{dish.name}</span>
                    </div>
                    <div className="text-sm text-gray-600 ml-16 mt-1">
                      {dish.quantity} × {dish.price?.toLocaleString('vi-VN')}k each
                    </div>
                  </div>
                  <div className="text-right">
                    <div className="font-bold text-green-600">
                      {dish.subtotal?.toLocaleString('vi-VN')}k
                    </div>
                  </div>
                </div>
              </div>
            ))}
          </div>
          <div className="px-4 py-3 bg-gray-100 border-t flex justify-between items-center">
            <span className="font-medium text-gray-900">Total</span>
            <span className="font-bold text-lg text-green-600">
              {order.total_bill?.toLocaleString('vi-VN')}k
            </span>
          </div>
        </div>
      </div>

      {order.notes && (
        <div>
          <h3 className="text-lg font-semibold text-gray-900 mb-2">Order Notes</h3>
          <div className="p-3 bg-blue-50 border border-blue-200 rounded-lg">
            <p className="text-blue-800">{order.notes}</p>
          </div>
        </div>
      )}
    </div>
  );
};

// Export default object with component names as keys
// This matches the LangGraph generative UI pattern
export default {
  multi_dish_lookup: MultiDishLookupResult,
  menu_value_count_and_price: MenuValueCountAndPriceResult,
  take_order: TakeOrderResult,
};
