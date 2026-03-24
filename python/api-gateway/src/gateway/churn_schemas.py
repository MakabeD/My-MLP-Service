from enum import Enum

from pydantic import BaseModel, Field


class PlanStatus(str, Enum):
    """Enum to restrict inputs to specific boolean-like string variants."""

    YES = "Yes"
    NO = "No"


class ChurnInput(BaseModel):
    """
    Input schema for the Churn prediction model.
    Validates data types and prevents impossible negative values
    in the telecommunications business domain.
    """

    Accountlength: int = Field(
        ..., gt=0, description="Customer's account length in months."
    )
    International_plan: PlanStatus = Field(
        ...,
        description="Does the customer have an international plan? ('Yes', 'No', 'True', or 'False')",
    )
    Voice_mail_plan: PlanStatus = Field(
        ...,
        description="Does the customer have a voice mail plan? ('Yes', 'No', 'True', or 'False')",
    )
    Number_vmail_messages: int = Field(
        ..., ge=0, description="Total number of voice mail messages received."
    )

    # --- Day Traffic ---
    Total_day_minutes: float = Field(
        ..., ge=0, description="Total minutes of day calls."
    )
    Total_day_calls: int = Field(..., ge=0, description="Total number of day calls.")
    Total_day_charge: float = Field(
        ..., ge=0, description="Total charge for day calls."
    )

    # --- Evening Traffic ---
    Total_eve_minutes: float = Field(
        ..., ge=0, description="Total minutes of evening calls."
    )
    Total_eve_calls: int = Field(
        ..., ge=0, description="Total number of evening calls."
    )
    Total_eve_charge: float = Field(
        ..., ge=0, description="Total charge for evening calls."
    )

    # --- Night Traffic ---
    Total_night_minutes: float = Field(
        ..., ge=0, description="Total minutes of night calls."
    )
    Total_night_calls: int = Field(
        ..., ge=0, description="Total number of night calls."
    )
    Total_night_charge: float = Field(
        ..., ge=0, description="Total charge for night calls."
    )

    # --- International Traffic ---
    Total_intl_minutes: float = Field(
        ..., ge=0, description="Total minutes of international calls."
    )
    Total_intl_calls: int = Field(
        ..., ge=0, description="Total number of international calls."
    )
    Total_intl_charge: float = Field(
        ..., ge=0, description="Total charge for international calls."
    )

    # --- Customer Service ---
    Customer_service_calls: int = Field(
        ..., ge=0, description="Number of calls made to customer service."
    )

    # Configuration block to generate an automatic test JSON in Swagger UI
    model_config = {
        "json_schema_extra": {
            "example": {
                "Accountlength": 117,
                "International_plan": "No",
                "Voice_mail_plan": "No",
                "Number_vmail_messages": 0,
                "Total_day_minutes": 184.5,
                "Total_day_calls": 97,
                "Total_day_charge": 31.37,
                "Total_eve_minutes": 351.6,
                "Total_eve_calls": 80,
                "Total_eve_charge": 29.89,
                "Total_night_minutes": 215.8,
                "Total_night_calls": 90,
                "Total_night_charge": 9.71,
                "Total_intl_minutes": 8.7,
                "Total_intl_calls": 4,
                "Total_intl_charge": 2.35,
                "Customer_service_calls": 1,
            }
        }
    }


class ChurnOutput(BaseModel):
    """
    Output schema returned by the API with the prediction results.
    """

    prediction: str = Field(
        ..., description="Final predicted label ('churn' or 'no churn')."
    )
    probability: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Mathematical probability of customer churn (from 0.0 to 1.0).",
    )
