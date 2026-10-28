"""Local standalone SQLPilot UI. Run with streamlit, never import ML runtimes."""
import asyncio
import os
from pathlib import Path

import streamlit as st

from citypulse_ai.sqlpilot.client import connect_tools
from citypulse_ai.sqlpilot.graph import run_agent
from citypulse_ai.sqlpilot.providers import ProviderConfig, configured_backend, schema_identity

DEFAULT_DATABASE = 'outputs/historical/samples/2024-01-n1000/citypulse.duckdb'


async def inspect_schema(database):
    async with connect_tools(database) as tools:
        return await tools.call('sqlpilot_schema', {})


def submit(question, database):
    return asyncio.run(run_agent(question, database, configured_backend()))


def main():
    st.set_page_config(page_title='CityPulse SQLPilot', layout='centered')
    st.title('CityPulse SQLPilot')
    st.caption('Ask about accepted recorded Citi Bike starts in the approved sample.')
    status = ProviderConfig.environment().status()
    if st.session_state.get('sqlpilot_result', {}).get('live_llm_exercised'):
        status['message'] = 'Live generation succeeded in the last invocation; this is not a continuous provider health check'
    st.info(status['message'])
    st.caption('Provider: '+status['provider']+' · Model: '+status['model'])
    database = os.getenv('SQLPILOT_DATABASE', DEFAULT_DATABASE)
    if not Path(database).is_file(): st.warning('The operator-configured sample database is unavailable.')
    if st.button('Inspect permitted schema'):
        try:
            schema = asyncio.run(inspect_schema(database))
            st.session_state['sqlpilot_schema'] = schema
        except Exception:
            st.error('MCP schema discovery failed. Check the approved sample and local process setup.')
    if 'sqlpilot_schema' in st.session_state:
        with st.expander('Permitted sample schema'):
            st.json(st.session_state['sqlpilot_schema'])
            st.code(schema_identity(st.session_state['sqlpilot_schema']['schema']), language=None)
    with st.form('question'):
        question = st.text_area('Analytical question', max_chars=2000,
                                placeholder='How many accepted starts are recorded in this sample?')
        asked = st.form_submit_button('Ask SQLPilot')
    if asked:
        with st.spinner('Discovering schema and checking the query…'):
            try: st.session_state['sqlpilot_result'] = submit(question, database)
            except Exception:
                st.session_state['sqlpilot_result'] = {'status': 'error', 'error': 'Application invocation failed safely'}
    if 'sqlpilot_result' not in st.session_state: return
    value = st.session_state['sqlpilot_result']
    if value.get('sql'):
        st.subheader('Generated SQL'); st.code(value['sql'], language='sql')
    if value['status'] == 'executed':
        result = value['result']
        st.subheader('Query results')
        st.dataframe([dict(zip(result['columns'], row)) for row in result['rows']], hide_index=True)
        st.text(value['explanation'])
        if result['truncated']: st.warning('Showing the first 100 rows; the result is truncated.')
    elif value['status'] == 'clarification':
        st.warning('Clarification needed'); st.text(value['clarification'])
    else: st.error(value.get('error', 'No result was produced'))
    st.caption('Live model generation verified for this invocation: '+str(value.get('live_llm_exercised', False)))
    with st.expander('Agent steps and MCP calls'):
        st.json({'steps': value.get('trace', []), 'mcp': value.get('mcp_trace', []),
                 'latency_seconds': value.get('latency_seconds')})
    st.caption('This sample does not establish full-history totals, physical demand or station availability.')


if __name__ == '__main__': main()
