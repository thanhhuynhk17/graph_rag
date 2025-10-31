# LangGraph Generative UI Components for MCP Tools

Components designed for LangGraph Platform's generative UI system to render MCP server tool results.

## Overview

This module provides React components that follow LangGraph's generative UI patterns. Components are automatically bundled by LangGraph Platform and loaded via `LoadExternalComponent` on the client side.

## Components

### Available Components
- `multi_dish_lookup` - Displays dish search results with found/missing/error handling
- `menu_value_count_and_price` - Renders organized menu with price analysis
- `take_order` - Shows complete order confirmation with customer details

## Configuration

### 1. langgraph.json

Add UI configuration pointing to this file:

```json
{
  "node_version": "20",
  "graphs": {
    "agent": "./src/agent/index.ts:graph"
  },
  "ui": {
    "mcp": "./src/ui/generative_ui.tsx"
  }
}
```

### 2. How Components are Props

Components receive the exact data structure that your MCP tools return in `structured_content`. For example:

```typescript
// From multi_dish_lookup tool
const props = {
  found_dishes: [...],
  missing_ids: [...],
  errors: [...],
  total_requested: 5,
  total_found: 3
};
```

## Usage in LangGraph Node

### Python Graph (if you add LangGraph integration)

```python
from langgraph.graph.ui import push_ui_message

@mcp.tool("multi_dish_lookup")
async def multi_dish_lookup(dish_ids: List[str]):
    result = run_multi_dish_lookup(dish_ids)

    # Push UI component - this will render on client via LoadExternalComponent
    push_ui_message("multi_dish_lookup", result.structured_content)

    return result
```

### JavaScript Graph

```javascript
import { typedUi } from "@langchain/langgraph-sdk/react-ui/server";

const ui = typedUi({
  multi_dish_lookup: (props) => null, // types will be inferred
  menu_value_count_and_price: (props) => null,
  take_order: (props) => null
});

// Push UI from graph node
ui.push("multi_dish_lookup", result.structured_content);
```

## Client-Side Usage

### Basic LoadExternalComponent

```tsx
import { LoadExternalComponent } from "@langchain/langgraph-sdk/react-ui";

<LoadExternalComponent
  stream={thread}
  message={uiMessage}
  namespace="mcp"
/>
```

### With Custom Properties

```tsx
<LoadExternalComponent
  stream={thread}
  message={uiMessage}
  namespace="mcp"
  meta={{ customProp: "value" }}
  fallback={<div>Loading MCP result...</div>}
/>
```

### Integration with Stream

```tsx
import { useStream } from "@langchain/langgraph-sdk/react";

const { values } = useStream({
  apiUrl: "http://localhost:2024",
  assistantId: "mcp-assistant"
});

// Render UI messages associated with each conversation message
{messages.map(message => (
  <div key={message.id}>
    {message.content}
    {values.ui
      ?.filter(ui => ui.metadata?.message_id === message.id)
      .map(ui => (
        <LoadExternalComponent
          key={ui.id}
          stream={values}
          message={ui}
          namespace="mcp"
        />
      ))}
  </div>
))}
```

## Component Props Structure

### MultiDishLookupProps
```typescript
{
  found_dishes: {
    dish_id: string;
    name_of_food: string;
    type_of_food: string;
    current_price: number;
    main_ingredients: string;
    taste: string;
  }[];
  missing_ids: string[];
  errors: { dish_id: string; error: string }[];
  total_requested: number;
  total_found: number;
}
```

### MenuValueCountAndPriceProps
```typescript
{
  menu_groups: Record<string, {
    type_name: string;
    count: number;
    dishes: {
      dish_id: string;
      name_of_food: string;
      current_price: number;
      type_of_food: string;
    }[];
  }>;
  total_dishes: number;
  total_groups: number;
  timestamp: string;
}
```

### TakeOrderProps
```typescript
{
  order: {
    order_id: string;
    customer_id: string;
    customer_name: string;
    total_bill: number;
    is_takeaway: boolean;
    table_id?: string;
    notes?: string;
    created_at?: string;
    ordered_dishes: {
      dish_id: string;
      name: string;
      quantity: number;
      price: number;
      subtotal: number;
    }[];
  };
  customer: {
    customer_id: string;
    full_name: string;
    phone?: string;
    email?: string;
  };
  status: "success" | "failed";
  timestamp: string;
}
```

## Dependencies

Install in your LangGraph Platform project:

```json
{
  "dependencies": {
    "@langchain/langgraph-sdk": "^latest",
    "tailwindcss": "^3.0.0"
  }
}
```

## Features

- ✅ **LangGraph Compatible** - Built for `LoadExternalComponent` usage
- ✅ **Vietnamese Localization** - VND currency, Vietnamese date formatting
- ✅ **Responsive Design** - Mobile-first with Tailwind CSS
- ✅ **Error Handling** - Graceful fallbacks for missing data
- ✅ **TypeScript Support** - Full type safety for MCP tool results
- ✅ **Shadow DOM Isolation** - Components loaded in isolated shadow DOM

## Development

### Running Locally

```bash
# Install dependencies
npm install

# Build for production
npm run build
```

### Hot Reload with LangGraph Platform

Components are automatically rebuilt when you make changes. LangGraph Platform will reload the bundled components.

## Integration Steps

1. **Copy Components**: Copy `generative_ui.tsx` to your LangGraph Platform project
2. **Configure UI**: Add to `langgraph.json` UI section
3. **Push UI Messages**: Modify your MCP tools to push UI messages
4. **Render on Client**: Use `LoadExternalComponent` in your React app

## Example Complete Integration

### MCP Tool with UI Push (Python)
```python
@mcp.tool("take_order")
def take_order(customer_id: str, full_name: str, dishes: List[Dict]):
    result = process_order(customer_id, full_name, dishes)

    # Push UI component for rich display
    push_ui_message("take_order", result.structured_content)

    return result
```

### Client Rendering
```tsx
{messages.map(message => (
  <div key={message.id}>
    <MessageBubble message={message} />

    {/* Render MCP tool UI components */}
    {values.ui?.filter(ui => ui.metadata?.message_id === message.id)
      .map(ui => (
        <LoadExternalComponent
          key={ui.id}
          stream={values}
          message={ui}
          namespace="mcp"
        />
      ))}
  </div>
))}
```

This creates a seamless chat experience where MCP tool results are automatically rendered as rich, interactive UI components instead of plain text responses.
