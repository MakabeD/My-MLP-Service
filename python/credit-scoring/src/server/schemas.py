from enum import Enum

from pydantic import BaseModel, Field


class sexEnum(Enum):
    male = "male"
    female = "female"


class housingEnum(Enum):
    own = "own"
    free = "free"
    rent = "rent"


class saving_accountsEnum(Enum):
    little = "little"
    na = "NA"
    quite_rich = "quite rich"
    rich = "rich"
    moderate = "moderate"


class checking_accountEnum(Enum):
    little = "little"
    moderate = "moderate"
    na = "NA"
    rich = "rich"


class PurposeEnum(Enum):
    car = "car"
    furniture_equipment = "furniture/equipment"
    radio_tv = "radio/TV"
    domestic_appliances = "domestic appliances"
    repairs = "repairs"
    education = "education"
    business = "business"
    vacation_others = "vacation/others"


class CreditScoringInput(BaseModel):
    """
    Define the structure of the input data for prediction.
    Field names must match the columns in the original dataset.
    """

    Age: int = Field(..., ge=1, description="Edad del solicitante en años.")
    Sex: sexEnum = Field(..., description="Sexo del solicitante.")
    Job: int = Field(..., ge=0, le=3, description="Nivel de habilidad laboral (0-3).")
    Housing: housingEnum = Field(..., description="Tipo de vivienda.")
    Saving_accounts: saving_accountsEnum = Field(
        ..., alias="Saving accounts", description="Estado de la cuenta de ahorros."
    )
    Checking_account: checking_accountEnum = Field(
        ..., alias="Checking account", description="estado de la cuenta corriente."
    )
    Credit_amount: int = Field(
        ..., gt=0, alias="Credit amount", description="Monto del credito solicitado."
    )
    Duration: int = Field(..., gt=0, description="Duracion del credito en meses.")
    Purpose: PurposeEnum = Field(..., description="Proposito del credito.")
    model_config = {
        "populate_by_name": True,
        "json_schema_extra": {
            "example": {
                "Age": 35,
                "Sex": "male",
                "Job": 1,
                "Housing": "free",
                "Saving accounts": "NA",
                "Checking account": "NA",
                "Credit amount": 9055,
                "Duration": 36,
                "Purpose": "education",
            }
        },
    }


class CreditScoringOutPut(BaseModel):
    """
    Define API response structure.
    """

    prediction: str = Field(..., description="Prediccion del riesgo('good' o 'bad')")
    probability: float = Field(..., description="Probabilidad de prediccion positiva")
