// Types for MCP Tool Results
export interface ToolResult {
  content: TextContent[];
  structured_content: any;
}

export interface TextContent {
  type: "text";
  text: string;
}

// Multi Dish Lookup Types
export interface DishLookupData {
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

export interface MultiDishLookupResult {
  found_dishes: DishLookupData[];
  missing_ids: string[];
  errors: { dish_id: string; error: string }[];
  total_requested: number;
  total_found: number;
}

// Menu Value Count and Price Types
export interface DishMenuItem {
  dish_id: string;
  name_of_food: string;
  current_price: number;
  type_of_food: string;
}

export interface MenuGroup {
  type_name: string;
  count: number;
  dishes: DishMenuItem[];
}

export interface MenuValueCountAndPriceResult {
  menu_groups: Record<string, MenuGroup>;
  total_dishes: number;
  total_groups: number;
  timestamp: string;
}

// Take Order Types
export interface OrderedDish {
  dish_id: string;
  name: string;
  quantity: number;
  price: number;
  subtotal: number;
}

export interface OrderData {
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

export interface CustomerData {
  customer_id: string;
  full_name: string;
  phone?: string;
  email?: string;
}

export interface TakeOrderResult {
  order: OrderData;
  customer: CustomerData;
  status: "success" | "failed";
  timestamp: string;
}

// Error result type
export interface ErrorResult {
  error: string;
  status: "failed";
  timestamp: string;
}
