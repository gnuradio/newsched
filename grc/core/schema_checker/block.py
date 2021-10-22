from .utils import Spec, expand

PARAM_SCHEME = expand(
    base_key=str,   # todo: rename/remove

    id=str,
    label=str,
    category=str,

    dtype=str,
    default=object,

    options=list,
    option_labels=list,
    option_attributes=Spec(types=dict, required=False, item_scheme=(str, list)),

    hide=str,
)
PORT_SCHEME = expand(
    label=str,
    domain=str,

    id=str,
    dtype=str,
    vlen=(int, str),

    multiplicity=(int, str),
    optional=(bool, int, str),
    hide=(bool, str),
)
TEMPLATES_SCHEME = expand(
    imports=str,
    var_make=str,
    var_value=str,
    make=str,
    callbacks=list,
)
CPP_TEMPLATES_SCHEME = expand(
    includes=list,
    declarations=str,
    make=str,
    var_make=str,
    callbacks=list,
    link=list,
    packages=list,
    translations=dict,
)
BLOCK_SCHEME = expand(
    id=Spec(types=str, required=True, item_scheme=None),
    label=str,
    category=str,
    flags=(list, str),

    parameters=Spec(types=list, required=False, item_scheme=PARAM_SCHEME),
    inputs=Spec(types=list, required=False, item_scheme=PORT_SCHEME),
    outputs=Spec(types=list, required=False, item_scheme=PORT_SCHEME),

    asserts=(list, str),
    value=str,

    templates=Spec(types=dict, required=False, item_scheme=TEMPLATES_SCHEME),
    cpp_templates=Spec(types=dict, required=False, item_scheme=CPP_TEMPLATES_SCHEME),

    documentation=str,
    grc_source=str,

    file_format=Spec(types=int, required=True, item_scheme=None),

    block_wrapper_path=str,  # todo: rename/remove
)

NEWSCHED_KEY_SCHEME = expand(
    id=str,
    type=str,
    options=list,
)

NEWSCHED_PROPERTY_SCHEME = expand(
    id=str,
    label=str,
    category=str,
    value=str,
    keys=Spec(types=list, required=False, item_scheme=NEWSCHED_KEY_SCHEME),
)

NEWSCHED_PARAM_SCHEME = expand(
    id=str,
    label=str,
    dtype=str,
    default=object,
    settable=(str,bool),
    gettable=(str,bool),

    options=list,
    option_labels=list,
    option_attributes=Spec(types=dict, required=False, item_scheme=(str, list)),

    hide=str,
)
NEWSCHED_PORT_SCHEME = expand(
    label=str,
    domain=str,
    direction=str,
    id=str,
    type=str,
    dims=str,
    size=str,

    multiplicity=(int, str),
    optional=(bool, int, str),
    hide=(bool, str),
)
NEWSCHED_IMPL_SCHEME = expand(
    id=str
)

NEWSCHED_BLOCK_SCHEME = expand(
    block=Spec(types=str, required=True, item_scheme=None),
    module=Spec(types=str, required=True, item_scheme=None),
    label=str,
    category=str,
    flags=(list, str),

    parameters=Spec(types=list, required=False, item_scheme=NEWSCHED_PARAM_SCHEME),
    properties=Spec(types=list, required=False, item_scheme=NEWSCHED_PROPERTY_SCHEME),
    ports=Spec(types=list, required=False, item_scheme=NEWSCHED_PORT_SCHEME),

    asserts=(list, str),
    value=str,

    templates=Spec(types=dict, required=False, item_scheme=TEMPLATES_SCHEME),
    cpp_templates=Spec(types=dict, required=False, item_scheme=CPP_TEMPLATES_SCHEME),

    documentation=str,
    grc_source=str,

    implementations=Spec(types=list, required=True, item_scheme=NEWSCHED_IMPL_SCHEME),

    file_format=Spec(types=int, required=True, item_scheme=None),

    block_wrapper_path=str,  # todo: rename/remove
)

