from typing import Dict

from thinker_ai.agent.role import Role


class Organization:
    roles: dict[str, Role] = {}

    def __init__(self,org_id:str, customer_id: str,  name: str, roles: dict[str, Role]):
        self.customer_id = customer_id
        self.org_id = org_id
        self.name = name
        self.roles = roles

    def add_role(self, role: Role):
        self.roles[role.name] = role


class CompositeOrganization(Organization):
    sub_organizations: Dict[str, Organization] = {}

    def __init__(self, org_id,str,customer_id:str,  name: str,roles: dict[str, Role]):
        super().__init__(org_id,customer_id,name,roles)

    def add_organization(self, sub_organization: Organization):
        if self.customer_id != sub_organization.customer_id:
            raise "must in the same customer"
        self.sub_organizations[sub_organization.org_id] = sub_organization
